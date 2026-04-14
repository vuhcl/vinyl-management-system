# Push to GitHub

The repo is initialized with the skeletal structure on **main** and four component branches: **recommender**, **nlp-classifier**, **price-estimator**, **web**.

## Steps

### 1. Create a new repository on GitHub

- Go to [github.com/new](https://github.com/new).
- Name it (e.g. `vinyl-management-system`).
- Leave it **empty** (no README, .gitignore, or license).
- Create the repository.

### 2. Add the remote and push

Replace `YOUR_USERNAME` and repo name if different:

**SSH:**
```bash
cd /path/to/vinyl_management_system
git remote add origin git@github.com:vuhcl/vinyl-management-system.git
```

**HTTPS:**
```bash
git remote add origin https://github.com/vuhcl/vinyl-management-system.git
```

### 3. Push all branches

```bash
git push -u origin main
git push origin recommender nlp-classifier price-estimator web
```

Or use the script (after creating the repo and setting the URL):

```bash
chmod +x scripts/push-to-github.sh
./scripts/push-to-github.sh git@github.com:YOUR_USERNAME/vinyl-management-system.git
```

## Branches

| Branch | Purpose |
|--------|---------|
| **main** | Default; full skeletal structure and integration |
| **recommender** | Recommender component work |
| **nlp-classifier** | NLP condition classifier work |
| **price-estimator** | Price estimator component work |
| **web** | Web interface work |

Each component branch starts from the same initial commit as `main`. Switch with `git checkout <branch>`.
