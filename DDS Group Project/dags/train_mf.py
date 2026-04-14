import json
import os
import tempfile
from datetime import datetime, timedelta

from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG

BUCKET = "aoty_data"
RATINGS_PATH = "user_ratings.tsv"
USER_EMB_PATH = "embeddings/user_embeddings.npy"
ITEM_EMB_PATH = "embeddings/item_embeddings.npy"
USER_MAP_PATH = "embeddings/user_id_mapping.json"
ITEM_MAP_PATH = "embeddings/item_id_mapping.json"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def train_mf(**context):
    import gcsfs
    import numpy as np
    import pandas as pd
    import torch
    from airflow.providers.google.cloud.hooks.gcs import GCSHook
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, TensorDataset

    # -------------------------------------------------------------------------
    # Load ratings from GCS
    # -------------------------------------------------------------------------
    gcs_hook = GCSHook(gcp_conn_id="google_cloud_default")
    credentials, project_id = gcs_hook.get_credentials_and_project_id()
    fs = gcsfs.GCSFileSystem(project=project_id, token=credentials)

    with fs.open(f"{BUCKET}/{RATINGS_PATH}") as f:
        user_ratings = pd.read_csv(f, sep="\t")

    print(f"Loaded {len(user_ratings):,} ratings")

    # -------------------------------------------------------------------------
    # Build ID mappings
    # -------------------------------------------------------------------------
    user_id_mapping: dict[str, int] = {
        username: idx for idx, username in enumerate(user_ratings["username"].unique())
    }
    item_id_mapping: dict[str, int] = {
        slug: idx for idx, slug in enumerate(user_ratings["slug"].unique())
    }

    user_ids = torch.tensor(
        user_ratings["username"].map(user_id_mapping).values, dtype=torch.long
    )
    item_ids = torch.tensor(
        user_ratings["slug"].map(item_id_mapping).values, dtype=torch.long
    )
    scores = torch.tensor(user_ratings["score"].values, dtype=torch.float)

    dataset = TensorDataset(user_ids, item_ids, scores)
    num_users = len(user_id_mapping)
    num_items = len(item_id_mapping)
    print(f"Users: {num_users:,}  Items: {num_items:,}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    class MF(nn.Module):
        def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)

        def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
            user_embeds = self.user_embedding(user_ids)
            item_embeds = self.item_embedding(item_ids)
            user_bias = self.user_bias(user_ids).squeeze()
            item_bias = self.item_bias(item_ids).squeeze()
            return torch.clamp(
                (user_embeds * item_embeds).sum(dim=1) + user_bias + item_bias, 0, 100
            )

    device = torch.device("cpu")
    embedding_dim = 128
    model = MF(num_users, num_items, embedding_dim).to(device)

    # -------------------------------------------------------------------------
    # Train / val / test split
    # -------------------------------------------------------------------------
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_subset, batch_size=20480, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=20480)
    test_loader = DataLoader(test_subset, batch_size=20480)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 10

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "best_model.pt")

        epochs = context["params"]["epochs"]
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_user_ids, batch_item_ids, batch_scores in train_loader:
                batch_user_ids = batch_user_ids.to(device)
                batch_item_ids = batch_item_ids.to(device)
                batch_scores = batch_scores.to(device)
                optimizer.zero_grad()
                preds = model(batch_user_ids, batch_item_ids)
                loss = loss_fn(preds, batch_scores)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_user_ids, batch_item_ids, batch_scores in val_loader:
                    batch_user_ids = batch_user_ids.to(device)
                    batch_item_ids = batch_item_ids.to(device)
                    batch_scores = batch_scores.to(device)
                    val_loss += loss_fn(
                        model(batch_user_ids, batch_item_ids), batch_scores
                    ).item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)
            print(
                f"Epoch {epoch + 1}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}"
                f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Evaluate on test set
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_user_ids, batch_item_ids, batch_scores in test_loader:
                batch_user_ids = batch_user_ids.to(device)
                batch_item_ids = batch_item_ids.to(device)
                batch_scores = batch_scores.to(device)
                test_loss += loss_fn(
                    model(batch_user_ids, batch_item_ids), batch_scores
                ).item()
        print(f"Test loss: {test_loss / len(test_loader):.4f}")

        # ---------------------------------------------------------------------
        # Extract and upload embeddings + mappings
        # ---------------------------------------------------------------------
        user_emb = model.user_embedding.weight.cpu().detach().numpy()
        item_emb = model.item_embedding.weight.cpu().detach().numpy()

        user_emb_local = os.path.join(tmpdir, "user_embeddings.npy")
        item_emb_local = os.path.join(tmpdir, "item_embeddings.npy")
        user_map_local = os.path.join(tmpdir, "user_id_mapping.json")
        item_map_local = os.path.join(tmpdir, "item_id_mapping.json")

        np.save(user_emb_local, user_emb)
        np.save(item_emb_local, item_emb)
        with open(user_map_local, "w") as f:
            json.dump(user_id_mapping, f)
        with open(item_map_local, "w") as f:
            json.dump(item_id_mapping, f)

        for local_path, gcs_path in [
            (user_emb_local, USER_EMB_PATH),
            (item_emb_local, ITEM_EMB_PATH),
            (user_map_local, USER_MAP_PATH),
            (item_map_local, ITEM_MAP_PATH),
        ]:
            gcs_hook.upload(
                bucket_name=BUCKET,
                object_name=gcs_path,
                filename=local_path,
            )
            print(f"Uploaded gs://{BUCKET}/{gcs_path}")

    context["ti"].xcom_push(key="user_map_gcs", value=USER_MAP_PATH)
    context["ti"].xcom_push(key="item_map_gcs", value=ITEM_MAP_PATH)


def store_mappings(**context):
    import json

    import gcsfs
    from airflow.providers.google.cloud.hooks.gcs import GCSHook
    from airflow.sdk.bases.hook import BaseHook
    from pymongo import MongoClient

    ti = context["ti"]
    user_map_object = ti.xcom_pull(task_ids="train_mf", key="user_map_gcs")
    item_map_object = ti.xcom_pull(task_ids="train_mf", key="item_map_gcs")

    gcs_hook = GCSHook(gcp_conn_id="google_cloud_default")
    credentials, project_id = gcs_hook.get_credentials_and_project_id()
    fs = gcsfs.GCSFileSystem(project=project_id, token=credentials)

    with fs.open(f"{BUCKET}/{user_map_object}") as f:
        user_id_mapping: dict[str, int] = json.load(f)
    with fs.open(f"{BUCKET}/{item_map_object}") as f:
        item_id_mapping: dict[str, int] = json.load(f)

    conn = BaseHook.get_connection("mongo_default")
    client = MongoClient(f"mongodb://{conn.host}:{conn.port}")
    db = client[conn.schema]

    for collection_name, mapping in [
        ("user_id_mapping", user_id_mapping),
        ("item_id_mapping", item_id_mapping),
    ]:
        collection = db[collection_name]
        collection.drop()
        collection.insert_many([{"key": k, "index": v} for k, v in mapping.items()])
        print(f"Stored {len(mapping):,} documents in {collection_name}")

    client.close()


with DAG(
    "train_mf",
    default_args=default_args,
    description="Train Matrix Factorization model, upload embeddings to GCS, store mappings in MongoDB",
    schedule=timedelta(weeks=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "spotify"],
    params={"epochs": 100},  # type: ignore[arg-type]
) as dag:
    task_train = PythonOperator(
        task_id="train_mf",
        python_callable=train_mf,
    )

    task_store = PythonOperator(
        task_id="store_mappings",
        python_callable=store_mappings,
    )

    task_train >> task_store
