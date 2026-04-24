# Guardian ML — GitHub Pages (Static Mode)

This folder contains the **fully self-contained static deployment** of Guardian ML.  
No backend, no server, no installation required.

---

## How to Deploy on GitHub Pages

### Step 1 — Set the base tag

Edit `index.html` line 9:

```html
<base href="/YOUR-REPO-NAME/" />
```

Replace `YOUR-REPO-NAME` with your actual GitHub repository name.

### Step 2 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial Guardian ML deploy"
git remote add origin https://github.com/USERNAME/YOUR-REPO-NAME.git
git push -u origin main
```

### Step 3 — Enable GitHub Pages

1. Go to your repo → **Settings** → **Pages**
2. Source: `main` branch → `/` (root) or `/github-pages` subfolder
3. Save and wait ~60 seconds

Your app will be live at:  
`https://USERNAME.github.io/YOUR-REPO-NAME/`

---

## How It Works Offline / Locally

Open `index.html` directly in any browser:

```bash
# macOS
open github-pages/index.html

# Linux
xdg-open github-pages/index.html

# Windows
start github-pages/index.html
```

The app loads `data/config.json` using a relative `fetch()` call — works on:
- `file://` protocol (local browser open)
- `localhost` (any static server)
- `https://username.github.io/repo-name/`

---

## Customizing the Simulation Data

Edit `data/config.json` to adjust:

| Key | Purpose |
|-----|---------|
| `risk_profiles` | Risk entities shown in the detection table |
| `model_metrics` | Simulated model performance values |
| `feature_importance` | Feature ranking for the importance chart |
| `trend_data` | Monthly trend line data |
| `system_stats` | Dashboard summary counts |
| `insights` | Insight bullets shown in the panel |
| `recommendations` | Action recommendations |

---

## File Structure

```
github-pages/
├── index.html          ← Main UI (contains <base> tag)
├── style.css           ← All styling (dark theme)
├── app.js              ← Simulation logic + Plotly charts
├── data/
│   └── config.json     ← All data (edit to customize)
└── README.md           ← This file
```

---

## No Build Tools Required

- Pure HTML, CSS, JavaScript
- Plotly.js loaded from CDN
- Zero npm, zero webpack, zero frameworks
