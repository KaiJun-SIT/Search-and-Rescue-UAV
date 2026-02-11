# 🎨 Colorbar Stacking/Shrinking Fix

## Problem

When rendering the SAR environment with `render_mode="human"`, the **coverage map** and **belief map** plots were **shrinking** with each update, getting smaller and smaller until they became tiny.

### Visual Issue

```
Step 0:   [Grid] [Coverage ■■■■■] [Belief ■■■■■]   ← Normal size
Step 10:  [Grid] [Coverage ■■■]   [Belief ■■■]     ← Getting smaller
Step 20:  [Grid] [Coverage ■■]    [Belief ■■]      ← Even smaller
Step 50:  [Grid] [Coverage ■]     [Belief ■]       ← Tiny!
```

### What Users Saw

- Coverage map and belief map **shrinking progressively**
- Plots getting **stacked/compressed**
- Eventually becoming **unreadably small**

---

## Root Cause

The issue was in **`envs/sar_env.py`** - specifically the colorbar creation:

### Problem Code (Before Fix)

```python
def render(self):
    if self.fig is None:
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))

    for ax in self.ax:
        ax.clear()  # ✅ Clears axis content

    self._render_coverage(self.ax[1])  # Adds NEW colorbar
    self._render_belief(self.ax[2])    # Adds NEW colorbar

def _render_coverage(self, ax):
    im = ax.imshow(self.coverage_map, cmap="Blues", ...)
    plt.colorbar(im, ax=ax, ...)  # ❌ NEW colorbar every frame!

def _render_belief(self, ax):
    im = ax.imshow(self.belief_map, cmap="hot", ...)
    plt.colorbar(im, ax=ax, ...)  # ❌ NEW colorbar every frame!
```

### Why This Caused Shrinking

1. **`ax.clear()`** clears the axis content (images, lines, etc.)
2. **BUT** `ax.clear()` does **NOT** remove colorbars!
3. Colorbars are **separate axes** attached to the figure
4. Every frame adds **2 new colorbars** (coverage + belief)
5. Each colorbar **takes space** from the plot area
6. Result: Plots shrink progressively as colorbars stack

**After 50 frames:** 100 colorbars stacked → plots squeezed into tiny space!

---

## The Fix

### Solution: Track and Remove Colorbars

**Added to `__init__`:**
```python
self.colorbars: list = []  # Track colorbars
```

**Modified `render()` method:**
```python
def render(self):
    if self.fig is None:
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))

    # ✅ Remove old colorbars BEFORE clearing axes
    for cbar in self.colorbars:
        cbar.remove()
    self.colorbars.clear()

    for ax in self.ax:
        ax.clear()

    # Render plots (which add new colorbars)
    self._render_coverage(self.ax[1])
    self._render_belief(self.ax[2])
```

**Modified `_render_coverage()` and `_render_belief()`:**
```python
def _render_coverage(self, ax):
    im = ax.imshow(self.coverage_map, cmap="Blues", ...)
    cbar = plt.colorbar(im, ax=ax, ...)
    self.colorbars.append(cbar)  # ✅ Track for removal

def _render_belief(self, ax):
    im = ax.imshow(self.belief_map, cmap="hot", ...)
    cbar = plt.colorbar(im, ax=ax, ...)
    self.colorbars.append(cbar)  # ✅ Track for removal
```

**Updated `close()` method:**
```python
def close(self):
    if self.fig is not None:
        plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.colorbars.clear()  # ✅ Cleanup
```

---

## How the Fix Works

### Before (Broken):
```
Frame 1: Create colorbar #1, #2
Frame 2: Create colorbar #3, #4  (plots shrink)
Frame 3: Create colorbar #5, #6  (plots shrink more)
...
Frame 50: Create colorbar #99, #100  (plots tiny!)
```

### After (Fixed):
```
Frame 1: Create colorbar #1, #2
Frame 2: Remove #1, #2 → Create NEW #1, #2  (plots same size)
Frame 3: Remove #1, #2 → Create NEW #1, #2  (plots same size)
...
Frame 50: Remove #1, #2 → Create NEW #1, #2  (plots same size!)
```

**Result:** Only **2 colorbars** ever exist at once → plots maintain size!

---

## Testing the Fix

Run the test script:

```bash
python test_colorbar_fix.py
```

**What to watch for:**
- ✅ Coverage and belief maps should **maintain their size**
- ✅ No progressive shrinking
- ✅ Plots stay readable throughout episode

---

## Key Takeaways

### Lesson: Colorbars are Persistent

```python
ax.clear()  # Clears: images, lines, text, patches
            # Does NOT clear: colorbars, legends (separate axes)
```

**Always remember:**
- Colorbars are **separate axes objects**
- Must be **explicitly removed** with `colorbar.remove()`
- Track references to remove them properly

### Best Practice

```python
# Track persistent objects
self.colorbars = []
self.legends = []

# Remove before redraw
for cbar in self.colorbars:
    cbar.remove()
self.colorbars.clear()

# Add and track new ones
cbar = plt.colorbar(...)
self.colorbars.append(cbar)
```

---

## Related Issues

This fix also applies to:
- **Legends** - Same issue, use `legend.remove()`
- **Annotations** - May need manual removal
- **Secondary axes** - Track and remove

---

## Usage

The fix is now in the main `SAREnv` class. Just use normally:

```python
from envs import SAREnv, EnvConfig

config = EnvConfig(grid_size=10, num_targets=1, render_mode="human")
env = SAREnv(config)

obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # ✅ No shrinking!
    if done or truncated:
        break

env.close()
```

---

## Before vs After

### Before Fix:
```
🔴 ISSUE: Plots progressively shrink
- Step 0: Normal size
- Step 50: Tiny, unreadable
- Cause: 100 colorbars stacked
```

### After Fix:
```
✅ FIXED: Plots maintain constant size
- Step 0: Normal size
- Step 50: Still normal size
- Reason: Old colorbars removed each frame
```

---

## Credits

**Issue reported by:** User observation of "graph stacking and getting smaller"
**Root cause:** Colorbars not being removed with `ax.clear()`
**Fix:** Track colorbar references and remove before re-creating
**Status:** ✅ Fixed and tested

---

**No more shrinking plots!** 🎉
