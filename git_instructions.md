What I Did to Check Branch Differences
1. Check current branch status
git status
Purpose: See uncommitted changes, untracked files, which branch you're on
2. Check recent commits on current branch
git log --oneline -10
Purpose: See what commits exist on your current branch (yayati)
3. Switch to master and check its history
git checkout master
git log --oneline -10
Purpose: See what's in master, compare commit history
4. See differences between branches
git diff master..yayati --stat
Purpose: Statistical summary of changes (files changed, lines added/removed) Alternative detailed view:
git diff master..yayati                    # Full diff
git diff master..yayati --name-only        # Just file names
git diff master..yayati -- <specific_file> # Specific file diff
5. Check for divergence
git log master..yayati --oneline    # Commits in yayati but not master
git log yayati..master --oneline    # Commits in master but not yayati
Purpose: See if branches have diverged (parallel work)
6. Visualize branch history
git log --oneline --graph --all --decorate
Purpose: See branch structure visually
Good Practices for Merging (Step-by-Step)
Phase 1: Pre-Merge Preparation
Step 1: Clean your feature branch (yayati)
# Make sure you're on feature branch
git checkout yayati

# Check for uncommitted changes
git status

# If there are uncommitted changes:
# Option A: Commit them
git add <files>
git commit -m "Descriptive message"

# Option B: Stash them (if not ready)
git stash

# Option C: Discard them (if unwanted)
git restore <files>
Step 2: Make sure everything works
# Run tests, scripts, verify outputs
python 5_03_01_rf_model_with_grid_search.py
python 5_11_permutation_feature_importance.py
# etc.
Step 3: Update from remote (if working with others)
git fetch origin
git status  # Check if remote has updates
Step 4: Update master locally
git checkout master
git pull origin master  # Get latest changes from remote
Step 5: Check if merge will be clean
# Preview what will be merged
git diff master..yayati --stat

# Check for potential conflicts
git merge yayati --no-commit --no-ff
git merge --abort  # Cancel the preview merge
Phase 2: The Merge
Step 6: Perform the merge
# Make sure you're on master
git checkout master

# Merge yayati into master
git merge yayati -m "Merge yayati: Add RF GridSearchCV and improvements"
Merge strategies:
Fast-forward (default if no divergence): Just moves master pointer forward
--no-ff: Always create merge commit (preserves branch history)
--squash: Combine all commits into one (cleaner history)
Which to use?
Fast-forward: Fine for simple feature branches
--no-ff: Good for preserving branch history
--squash: Good if you have many messy commits you want to clean up
Step 7: Handle conflicts (if any)
# Git will tell you which files have conflicts
git status

# Edit conflicted files (look for <<<<<<, ======, >>>>>>)
# Choose which changes to keep

# After resolving:
git add <resolved_files>
git commit  # Completes the merge
Phase 3: Post-Merge Verification
Step 8: Verify the merge
# Check the merge commit
git log --oneline -5

# Verify everything still works
python 5_03_01_rf_model_with_grid_search.py

# Check what changed
git diff HEAD~1  # Compare to before merge
Step 9: Push to remote
git push origin master
Step 10: Clean up (optional)
# Delete merged branch locally (if done with it)
git branch -d yayati

# Delete from remote (if applicable)
git push origin --delete yayati

# Or keep it for continued work
Best Practices Summary
Before Merging:
✅ Commit all changes on feature branch
✅ Test everything works
✅ Update master from remote
✅ Review what's being merged (git diff master..yayati)
✅ Clean up experimental files/code
During Merging:
✅ Always merge FROM feature branch INTO master (not the reverse)
✅ Write descriptive merge commit messages
✅ Resolve conflicts carefully
✅ Use --no-ff if you want to preserve branch history
After Merging:
✅ Verify everything still works
✅ Push to remote immediately
✅ Clean up local branches if done
✅ Consider tagging important merges: git tag -a v1.0 -m "RF model release"
Useful Commands Cheatsheet
# Check differences between branches
git diff master..yayati --stat           # Summary
git log master..yayati --oneline         # Commits in yayati not in master

# Merge preview (dry run)
git merge yayati --no-commit --no-ff
git merge --abort

# Actual merge
git checkout master
git merge yayati

# If you mess up
git merge --abort                        # Cancel ongoing merge
git reset --hard HEAD~1                  # Undo last merge (DANGEROUS!)
git reflog                               # See all actions, can recover

# Clean merge with squash (all commits → 1)
git merge --squash yayati
git commit -m "Add RF GridSearchCV model"
Your Specific Case
Since you've fixed everything, here's what you should do:
# 1. Make sure yayati is clean
git checkout yayati
git status  # Should show clean working directory

# 2. Update master
git checkout master
git pull origin master  # (if working with remote)

# 3. Preview merge
git diff master..yayati --stat

# 4. Merge
git merge yayati -m "Merge yayati: Add RF GridSearchCV, refactor helpers, update models"

# 5. Test
python 5_03_01_rf_model_with_grid_search.py

# 6. Push
git push origin master

# 7. Optionally push yayati too
git push origin yayati