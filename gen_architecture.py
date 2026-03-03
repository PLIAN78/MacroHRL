"""
MacroHRL Architecture diagram — clean square-corner boxes, clean routing.
All connectors use simple horizontal or vertical lines with arrowheads at the end.
No upward arrows — all flow goes right or downward.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.5)
ax.axis('off')
fig.patch.set_facecolor('white')

C_INPUT  = '#D6EAF8'
C_PROC   = '#FDEBD0'
C_REGIME = '#D5F5E3'
C_META   = '#E8DAEF'
C_SUB    = '#FADBD8'
C_OUT    = '#EBF5FB'
C_REWARD = '#F8F9FA'
EDGE     = '#2C3E50'

def box(ax, x, y, w, h, label, sublabel=None, color='#FFFFFF', fontsize=9):
    rect = Rectangle((x, y), w, h, linewidth=1.4, edgecolor=EDGE,
                      facecolor=color, zorder=3)
    ax.add_patch(rect)
    cy = y + h/2 + (0.14 if sublabel else 0)
    ax.text(x + w/2, cy, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', zorder=4,
            multialignment='center')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel, ha='center', va='center',
                fontsize=7.5, color='#555555', zorder=4, style='italic',
                multialignment='center')

def arrow_h(ax, x1, y, x2):
    """Horizontal arrow."""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=EDGE, lw=1.3,
                                connectionstyle='arc3,rad=0'), zorder=5)

def arrow_v(ax, x, y1, y2):
    """Vertical arrow (downward)."""
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=EDGE, lw=1.3,
                                connectionstyle='arc3,rad=0'), zorder=5)

def line(ax, xs, ys):
    """Draw a polyline (no arrowhead)."""
    ax.plot(xs, ys, color=EDGE, lw=1.3, zorder=4, solid_capstyle='butt')

# ── BOX LAYOUT ──────────────────────────────────────────────────────────────
# Left column (inputs)
box(ax, 0.2, 5.5, 2.5, 1.2, 'Macro Indicators',
    '(CPI YoY, VIX,\nYield Curve)', C_INPUT)
box(ax, 0.2, 3.8, 2.5, 1.2, 'Historical Price Data',
    '(8 ETFs,\nYahoo Finance)', C_INPUT)
box(ax, 0.2, 1.9, 2.5, 1.2, 'CVaR-Penalized\nReward Function',
    'Eq. (1)', C_REWARD)

# Centre column
box(ax, 4.3, 4.9, 3.0, 1.3, 'Rule-Based\nRegime Classifier',
    '(CPI & VIX\nThresholds)', C_PROC)
box(ax, 4.3, 3.0, 3.0, 1.3, 'Regime Label',
    '(Bull, Bear,\nSideways, Crisis)', C_REGIME)

# Right column (HRL)
box(ax, 9.3, 5.1, 3.2, 1.3, 'Meta-Controller\n$\\pi_{meta}$',
    'Selects Active\nSub-Policy', C_META)
box(ax, 9.3, 3.1, 3.2, 1.3, 'Sub-Controller\n$\\pi_{sub}^{r_t}$',
    'Regime-Specific\nPortfolio Allocation', C_SUB)
box(ax, 9.3, 1.2, 3.2, 1.3, 'Portfolio Weights',
    '$w_t = [w_1, \\ldots, w_N]$', C_OUT)

# ── CONNECTORS ──────────────────────────────────────────────────────────────
# 1. Macro Indicators → Classifier  (right from mid of box, then straight right)
arrow_h(ax, 2.7, 6.1, 4.3)

# 2. Price Data → Classifier  (right from mid, elbow up to classifier level)
#    go right to x=3.8, then up to y=5.55, then right to classifier
line(ax, [2.7, 3.8], [4.4, 4.4])
line(ax, [3.8, 3.8], [4.4, 5.55])
arrow_h(ax, 3.8, 5.55, 4.3)

# 3. Classifier → Regime Label  (straight down)
arrow_v(ax, 5.8, 4.9, 4.3)

# 4. Regime Label → Meta-Controller  (right, elbow up to meta level)
line(ax, [7.3, 8.5], [3.65, 3.65])
line(ax, [8.5, 8.5], [3.65, 5.75])
arrow_h(ax, 8.5, 5.75, 9.3)

# 5. Macro Indicators → Meta-Controller  (long right path at top)
# Use a separate elbow at x=9.0 so it arrives at the Meta box top
line(ax, [2.7, 9.0], [6.1, 6.1])
line(ax, [9.0, 9.0], [6.1, 6.4])
arrow_h(ax, 9.0, 6.4, 9.3)

# 6. Meta-Controller → Sub-Controller  (straight down)
arrow_v(ax, 10.9, 5.1, 4.4)

# 7. Sub-Controller → Portfolio Weights  (straight down)
arrow_v(ax, 10.9, 3.1, 2.5)

# 8. Reward Function → Sub-Controller  (right, elbow up)
line(ax, [2.7, 8.5], [2.5, 2.5])
line(ax, [8.5, 8.5], [2.5, 3.65])
arrow_h(ax, 8.5, 3.65, 9.3)

# ── COLUMN HEADERS ──────────────────────────────────────────────────────────
ax.text(1.45, 7.1, 'Inputs', ha='center', fontsize=11,
        fontweight='bold', color='#1A5276')
ax.text(5.8, 7.1, 'Preprocessing', ha='center', fontsize=11,
        fontweight='bold', color='#784212')
ax.text(10.9, 7.1, 'HRL Framework', ha='center', fontsize=11,
        fontweight='bold', color='#512E5F')

# ── TIMESCALE LABELS REMOVED ─────────────────────────────────────────────────

# ── TITLE ───────────────────────────────────────────────────────────────────
ax.text(6.5, 7.4, 'MacroHRL Architecture', ha='center', fontsize=13,
        fontweight='bold', color='#2C3E50')

plt.savefig('/home/ubuntu/research/figures/fig7_architecture.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("Done.")
