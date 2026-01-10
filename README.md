#  Winter Hackathon - Repository Setup Guide

Welcome to the Winter Hackathon organized by **Secptix** and **GDG**! To track your progress and collaborate effectively, every team must fork the official repository and follow the steps below.

---

##  Fork the Repository

1. Go to the GitHub page: [https://github.com/gdgsjec/WinterHackathon-TEAM_NAME-](https://github.com/gdgsjec/WinterHackathon-TEAM_NAME-)
2. Click on the **"Fork"** button in the upper-right corner of the page
3. Assign the repository name as `WinterHackathon-<Team_Name>`

**Example:**
```
Repository Name: WinterHackathon-BossBandit
```

---

##  Clone Your Forked Repository

1. Go to your forked repository on GitHub
2. Click the green **"Code"** button, then click the clipboard icon to copy the URL
3. Open your terminal and run the following command to copy the repository to your local machine (replace `<repository-url>` with your forked repository URL):

```bash
git clone <repository-url>
```

---

##  Start Working on Your Project

Begin building your solution! Collaborate with your teammates and push changes regularly.

---

## Commit Your Changes

Track and save your progress using Git:

### Check the status of your changes
```bash
git status
```

### Stage your changes
Use the `git add` command to stage the changes you want to commit:
```bash
git add .
```

### Commit with a meaningful message

#### **Option 1: Simple Commit Format** (Beginner Friendly)
Use this if you're new to Git:
```bash
git commit -m "Your descriptive commit message"
```

#### **Option 2: Conventional Commits** (Recommended)
Follow this format for more structured, professional commit history:
```bash
git commit -m "<type>(<scope>): <subject>"
```

**Commit Types:**

| Type | Purpose |
|------|---------|
| `feat` | For a new feature for the user, not a new feature for build script. Such commit will trigger a release bumping a MINOR version |
| `fix` | For a bug fix for the user, not a fix to a build script. Such commit will trigger a release bumping a PATCH version |
| `perf` | For performance improvements. Such commit will trigger a release bumping a PATCH version |
| `docs` | For changes to the documentation |
| `test` | For adding missing tests, refactoring tests; no production code change |
| `style` | For formatting changes, missing semicolons, etc |
| `refactor` | For refactoring production code, e.g. renaming a variable |
| `build` | For updating build configuration, development tools or other changes irrelevant to the user |

- **Scope:** Area of change (e.g., api, ui, auth)
- **Subject:** One-line summary in present tense, no period at the end

**Example:**
```bash
git commit -m "fix(button): fix submit button not working"
```

---

## Push Your Changes

Send your local commits to GitHub:
```bash
git push origin
```

---

##  Tips for Success

- **Commit often:** Small, frequent commits help track progress and fix bugs easily
- **Write clear messages:** Describe what you did in each commit
- **Collaborate:** Make sure everyone in your team contributes
- **Stay organized:** Use branches for different features if needed
- **Test regularly:** Ensure your code works before pushing

---

##  Need Help?

For any issues or doubts, reach out to the organizing team.

**Happy Hacking!** ✨

---

*Organized by Secptix & GDG*  
