# 🚀 Quick Deploy Guide for GLD Price Distribution Analyzer

## Fastest Option: Streamlit Community Cloud (Free)

### Step 1: Push to GitHub
```bash
# If you haven't already created a GitHub repo:
gh repo create gld-analyzer --public --source=. --remote=origin --push
```

Or manually:
1. Create a new repo on [GitHub](https://github.com/new)
2. Name it `gld-analyzer` (or your preferred name)
3. Push your code:
```bash
git remote add origin https://github.com/YOUR-USERNAME/gld-analyzer.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select:
   - Repository: `YOUR-USERNAME/gld-analyzer`
   - Branch: `main`
   - Main file: `gld_app.py`
5. Click "Deploy!"

**Your app will be live in ~2 minutes!** 🎉

URL format: `https://YOUR-USERNAME-gld-analyzer.streamlit.app`

## Alternative: Deploy to Heroku

```bash
# Install Heroku CLI first, then:
heroku create gld-analyzer-YOUR-NAME
git push heroku main
heroku open
```

## Alternative: Deploy to Railway

1. Visit [railway.app](https://railway.app)
2. Click "Start a New Project"
3. Choose "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects Streamlit and deploys!

## Features Included

✅ Real-time GLD data fetching
✅ Interactive probability calculations
✅ Dark theme optimized
✅ Mobile responsive
✅ No API keys needed
✅ Auto-refreshing data cache

## Share Your App

Once deployed, share your app URL with anyone! They can:
- Model their own GLD price beliefs
- Calculate custom probability scenarios
- See real-time market data
- Export probability calculations

No sign-up or authentication required - it's completely open!