#  Winter Hackathon - Repository Setup Guide

Welcome to the Winter Hackathon organized by **Sceptix** and **GDG SJEC**! To track your progress and collaborate effectively, every team must fork the official repository and follow the steps below.

---

##  Fork the Repository

1. Go to the GitHub page: [https://github.com/gdgsjec/WinterHackathon-TEAM_NAME-](https://github.com/gdgsjec/WinterHackathon-TEAM_NAME-)
2. Click on the **"Fork"** button in the upper-right corner of the page
3. Assign the repository name as `WinterHackathon-<TeamName>`
> 📝 **Note:** Please use **[PascalCase](https://pascal-case.com/)** for your team name.


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
## Update the README (Mandatory)

Before writing any code, replace the existing README.md with information specific to your project.
Your README is the first thing mentors and judges will see, so make sure it clearly explains what you built and why.
You can replace the README with the recommended template below and update the contents as you work on the project.

### Recommended Format for README:
Use the structure below:
```
 # Project Name

## Description
Briefly explain what your project does, the problem it solves, and who it is for.

# Demo Video Link: <insert Google Drive link to the demo video of the working of your project>

## Features
- Key feature 1
- Key feature 2
- Key feature 3

## Tech Stack
List the technologies, frameworks, and tools used in the project.

## Google Technologies Used
> ⚠️ Using Google products is **mandatory** for this hackathon.

List the Google technologies you used and clearly explain **why** you chose them.

**Example:**
- **Firebase Authentication** – For secure and easy user authentication
- **Firebase Firestore** – To store and manage real-time data
- **Google Maps API** – To enable location-based features

## Setup Instructions
Steps to run the project locally:
1. Clone the repository
2. Install dependencies
3. Add environment variables (if any)
4. Run the project

## Team Members
- Name 1
- Name 2
- Name 3
```


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

*Organized by Sceptix & GDG SJEC*  
