"""CoverWise design tokens and global CSS (injected once per run).

Palette follows frontend/design/README.md: navy/ink text, white + cool-grey surfaces, one
cobalt accent, muted green for valid states, amber for the disclaimer. Body text on white
is #0B1F44 (contrast ≈ 16:1); muted text #475569 (≈ 7.5:1).
"""

TOKENS = {
    "navy": "#0B1F44", "ink": "#16294F", "cobalt": "#2F5BEA", "cobalt_dark": "#1F44C4",
    "bg": "#F4F6FB", "surface": "#FFFFFF", "border": "#E2E8F2", "muted": "#475569",
    "green": "#1E7F5C", "green_bg": "#E7F5EF", "amber": "#92590A", "amber_bg": "#FFF6E0",
    "amber_border": "#F2C66D", "red": "#B3261E", "red_bg": "#FDECEA",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --navy:#0B1F44; --ink:#16294F; --cobalt:#2F5BEA; --cobalt-dark:#1F44C4; --bg:#F4F6FB;
  --surface:#FFFFFF; --border:#E2E8F2; --muted:#475569; --green:#1E7F5C; --green-bg:#E7F5EF;
  --amber:#92590A; --amber-bg:#FFF6E0; --amber-border:#F2C66D; --red:#B3261E; --red-bg:#FDECEA;
  --radius:16px; --shadow:0 1px 2px rgba(11,31,68,.06), 0 8px 24px rgba(11,31,68,.06);
}
html, body, [class*="css"], .stApp{font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif !important;}
.stApp{background:var(--bg); color:var(--navy);}
#MainMenu, footer, header[data-testid="stHeader"]{visibility:hidden; height:0;}
.block-container{max-width:1180px; padding-top:1.2rem; padding-bottom:4rem;}
h1,h2,h3,h4{color:var(--navy) !important; letter-spacing:-.02em; font-weight:700;}
p, li, label, span{color:inherit;}

/* ---------- top navigation ---------- */
.cw-brand{display:flex; align-items:center; gap:10px; padding:6px 0;}
.cw-logo{width:34px; height:34px; border-radius:10px; background:linear-gradient(135deg,var(--navy),var(--cobalt));
  display:flex; align-items:center; justify-content:center; color:#fff; font-weight:800; font-size:18px;}
.cw-wordmark{font-size:22px; font-weight:800; color:var(--navy); letter-spacing:-.03em; line-height:1;}
.cw-sub{font-size:12px; color:var(--muted); margin-top:2px;}
div[data-testid="stHorizontalBlock"] .stButton>button{border-radius:999px; font-weight:600; padding:.45rem 1rem;
  border:1px solid transparent; background:transparent; color:var(--navy); box-shadow:none;}
div[data-testid="stHorizontalBlock"] .stButton>button:hover{background:#E8EEFC; color:var(--cobalt-dark); border-color:transparent;}
div[data-testid="stHorizontalBlock"] .stButton>button[kind="primary"]{background:var(--navy); color:#fff;}
.cw-navrule{height:1px; background:var(--border); margin:6px 0 22px;}

/* ---------- cards ---------- */
.cw-card{background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  padding:24px; box-shadow:var(--shadow); height:100%;}
.cw-card h3{margin:0 0 6px; font-size:18px;}
.cw-card p{color:var(--muted); margin:0; line-height:1.55; font-size:15px;}
.cw-icon{width:40px; height:40px; border-radius:12px; background:#E8EEFC; color:var(--cobalt-dark);
  display:flex; align-items:center; justify-content:center; font-weight:700; margin-bottom:14px;}
.cw-eyebrow{display:inline-block; font-size:12px; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
  color:var(--cobalt-dark); background:#E8EEFC; padding:5px 12px; border-radius:999px;}

/* ---------- hero ---------- */
.cw-hero{background:linear-gradient(135deg,#0B1F44 0%,#14357E 60%,#2F5BEA 120%); border-radius:24px;
  padding:56px 48px; color:#fff; box-shadow:var(--shadow);}
.cw-hero h1{color:#fff !important; font-size:44px; line-height:1.1; margin:14px 0 14px; max-width:720px;}
.cw-hero p{color:#DCE6FF; font-size:18px; line-height:1.55; max-width:640px; margin:0;}
.cw-hero .cw-eyebrow{background:rgba(255,255,255,.14); color:#fff;}
.cw-hero-notice{margin-top:22px; display:inline-block; background:rgba(255,246,224,.12); border:1px solid rgba(242,198,109,.55);
  color:#FFE9B5; padding:8px 14px; border-radius:12px; font-size:14px; font-weight:500;}

/* ---------- section titles ---------- */
.cw-h2{font-size:26px; font-weight:700; margin:34px 0 6px;}
.cw-lead{color:var(--muted); font-size:16px; margin:0 0 18px; max-width:760px; line-height:1.55;}

/* ---------- metric cards ---------- */
.cw-metric{background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:18px 20px; box-shadow:var(--shadow);}
.cw-metric .label{font-size:12px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--muted);}
.cw-metric .value{font-size:30px; font-weight:800; color:var(--navy); letter-spacing:-.02em; margin-top:6px; font-variant-numeric:tabular-nums;}
.cw-metric .hint{font-size:13px; color:var(--muted); margin-top:4px;}

/* ---------- estimate ---------- */
.cw-estimate{background:linear-gradient(160deg,#0B1F44,#17408F); color:#fff; border-radius:20px; padding:34px; box-shadow:var(--shadow);}
.cw-estimate .label{font-size:13px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#B9C9F5;}
.cw-estimate .amount{font-size:60px; font-weight:800; letter-spacing:-.03em; margin:8px 0 4px; font-variant-numeric:tabular-nums; color:#fff;}
.cw-estimate .unit{color:#DCE6FF; font-size:15px;}
.cw-placeholder{background:var(--surface); border:2px dashed #C6D2EA; border-radius:20px; padding:48px 28px; text-align:center; color:var(--muted);}
.cw-placeholder .big{font-size:40px; font-weight:800; color:#B6C3DD;}

/* ---------- banners & pills ---------- */
.cw-disclaimer{background:var(--amber-bg); border:1px solid var(--amber-border); color:var(--amber); border-radius:14px;
  padding:14px 16px; font-size:14.5px; font-weight:600; line-height:1.5;}
.cw-error{background:var(--red-bg); border:1px solid #F3B8B3; color:var(--red); border-radius:14px; padding:14px 16px; font-size:14.5px; line-height:1.5;}
.cw-error b{color:var(--red);}
.cw-pill{display:inline-block; padding:4px 12px; border-radius:999px; font-size:12.5px; font-weight:700;}
.cw-pill.ok{background:var(--green-bg); color:var(--green);}
.cw-pill.warn{background:var(--amber-bg); color:var(--amber);}
.cw-pill.info{background:#E8EEFC; color:var(--cobalt-dark);}

/* ---------- summary grid ---------- */
.cw-summary{display:grid; grid-template-columns:repeat(3,1fr); gap:12px;}
.cw-summary div{background:#F7F9FD; border:1px solid var(--border); border-radius:12px; padding:12px 14px;}
.cw-summary span{display:block; font-size:12px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.05em;}
.cw-summary b{font-size:17px; color:var(--navy);}

/* ---------- tables ---------- */
.cw-table-wrap{overflow-x:auto; border:1px solid var(--border); border-radius:var(--radius); background:var(--surface); box-shadow:var(--shadow);}
table.cw-table{width:100%; border-collapse:collapse; font-size:14.5px; font-variant-numeric:tabular-nums;}
.cw-table th{background:#F1F4FB; text-align:left; padding:12px 16px; font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted);}
.cw-table td{padding:13px 16px; border-top:1px solid var(--border); color:var(--navy);}
.cw-table tr.sel td{background:#EEF3FF; font-weight:600;}

/* ---------- forms ---------- */
div[data-testid="stForm"]{background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:26px; box-shadow:var(--shadow);}
div[data-testid="stForm"] label p{font-weight:600; color:var(--navy);}
div[data-baseweb="input"], div[data-baseweb="select"]>div{border-radius:12px !important;}
.stButton>button, .stFormSubmitButton>button{border-radius:12px; font-weight:700; padding:.6rem 1.4rem; border:1px solid var(--border);}
.stFormSubmitButton>button[kind="primaryFormSubmit"], .stButton>button[kind="primary"]{background:var(--cobalt); color:#fff; border-color:var(--cobalt);}
.stFormSubmitButton>button[kind="primaryFormSubmit"]:hover, .stButton>button[kind="primary"]:hover{background:var(--cobalt-dark); border-color:var(--cobalt-dark); color:#fff;}
.stButton>button:focus-visible, .stFormSubmitButton>button:focus-visible{outline:3px solid #9DB5FF; outline-offset:2px;}
.cw-hint{font-size:13px; color:var(--muted); margin:-6px 0 12px;}
.cw-foot{color:var(--muted); font-size:13px; margin-top:42px; text-align:center;}

@media (max-width: 800px){
  .cw-hero{padding:32px 22px;} .cw-hero h1{font-size:30px;} .cw-estimate .amount{font-size:42px;}
  .cw-summary{grid-template-columns:1fr 1fr;}
}
"""
