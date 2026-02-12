# Run this script to push without LFS (avoids GitHub LFS auth errors).
# Run from repo root: .\push-without-lfs.ps1

# 1. Undo last commit but keep all changes
git reset --soft HEAD~1

# 2. Uninstall LFS in this repo so it doesn't intercept push
git lfs uninstall

# 3. Remove large files from the index (they are in .gitignore now)
git rm --cached "nlp2021-hw1/model/best-gru2-parameters-.697.pt" 2>$null
git rm --cached "nlp2021-hw2/model/a1-epoch=06-val_F1=0.853.ckpt" 2>$null
git rm --cached "nlp2021-hw2/model/a2-epoch=16-val_F1=0.881.ckpt" 2>$null
git rm --cached "nlp2021-hw2/model/b-epoch=15-val_F1=0.601.ckpt" 2>$null
git rm --cached "nlp2021-hw2/model/c-epoch=11-val_F1=0.855.ckpt" 2>$null
git rm --cached "nlp2021-hw2/model/d-epoch=08-val_F1=0.614.ckpt" 2>$null
git rm --cached "nlp2021-hw2/logs/server.stdout" 2>$null
git rm --cached "nlp2021-hw2/logs/server.stderr" 2>$null
git rm --cached "nlp2021-hw3/model/wic-wsd-epoch=09-wic_val_Accuracy=0.67.ckpt" 2>$null
git rm --cached "nlp2021-hw3/logs/server.stdout" 2>$null
git rm --cached "nlp2021-hw3/logs/server.stderr" 2>$null

# 4. Stage everything else (including updated .gitignore and .gitattributes)
git add .

# 5. Commit
git commit -m "upload projects (code and data; large binaries excluded)"

# 6. Push (no LFS, so no batch auth)
Write-Host "Pushing to origin main..."
git push -u origin main
