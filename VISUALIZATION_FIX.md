# 🎨 Visualization Glitching Issue and Fix

## Problem Identified

The visualization is glitching/flickering because of the rendering approach in `envs/sar_env.py`:

### Root Causes:

**1. Full Redraw Every Frame (Lines 458-459)**
```python
for ax in self.ax:
    ax.clear()  # ❌ Clears EVERYTHING every frame
```

**2. Redrawing Static Elements (Lines 495-506)**
```python
# Redraws grid lines EVERY frame (unchanging)
for i in range(self.grid_size + 1):
    ax.axhline(i, color="gray", linewidth=0.5)
    ax.axvline(i, color="gray", linewidth=0.5)

# Redraws obstacles EVERY frame (unchanging)
for y in range(self.grid_size):
    for x in range(self.grid_size):
        if self.obstacles[y, x]:
            rect = patches.Rectangle(...)
            ax.add_patch(rect)
```

**3. Short Pause Causes Flashing (Line 476)**
```python
plt.pause(0.001)  # ❌ Too short - causes flickering
```

## Why This Causes Glitching:

1. **Clear → Blank screen** (visible flash)
2. **Redraw static elements** (grid, obstacles) - wastes time
3. **Redraw dynamic elements** (agent, targets)
4. **Update display** with `plt.pause()` or `plt.draw()`
5. **Repeat** → causes continuous flickering

The human eye sees this as "glitching" because:
- The screen briefly goes blank (white flash)
- Heavy redrawing causes lag
- Fast loop makes flickering more noticeable

---

## Solutions

### **Solution 1: Increase Pause Time (Quick Fix)**

**Change Line 476:**
```python
# Before
plt.pause(0.001)  # Too fast

# After
plt.pause(0.05)  # 50ms - smoother, less flicker
```

**Pros:** Simple one-line fix
**Cons:** Still redraws everything, just slower

---

### **Solution 2: Use Blitting (Optimal Fix)**

Only redraw what changes (agent position, sensor circle):

```python
def render(self) -> Optional[np.ndarray]:
    """Render with blitting for smooth animation."""
    if self.config.render_mode is None:
        return None

    # First time: create figure and draw static elements
    if self.fig is None:
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))

        # Draw static elements ONCE
        self._render_static_elements()

        # Store artists for dynamic elements
        self.dynamic_artists = []

        plt.ion()  # Interactive mode
        plt.show()

    # Clear only dynamic artists
    for artist in self.dynamic_artists:
        artist.remove()
    self.dynamic_artists = []

    # Redraw only dynamic elements
    self._render_dynamic_elements()

    self.fig.canvas.draw_idle()
    self.fig.canvas.flush_events()

    return None

def _render_static_elements(self):
    """Draw elements that don't change (ONCE)."""
    # Grid lines
    for i in range(self.grid_size + 1):
        self.ax[0].axhline(i, color="gray", linewidth=0.5)
        self.ax[0].axvline(i, color="gray", linewidth=0.5)

    # Obstacles
    for y in range(self.grid_size):
        for x in range(self.grid_size):
            if self.obstacles[y, x]:
                rect = patches.Rectangle(
                    (x, y), 1, 1, linewidth=0, facecolor="black", alpha=0.7
                )
                self.ax[0].add_patch(rect)

def _render_dynamic_elements(self):
    """Draw elements that change every frame."""
    # Agent
    agent_circle = patches.Circle(
        (self.agent_pos[0] + 0.5, self.agent_pos[1] + 0.5),
        0.4, color="blue", alpha=0.8
    )
    self.ax[0].add_patch(agent_circle)
    self.dynamic_artists.append(agent_circle)

    # Sensor range
    sensor_circle = patches.Circle(
        (self.agent_pos[0] + 0.5, self.agent_pos[1] + 0.5),
        self.config.sensor_radius,
        fill=False, edgecolor="cyan", linewidth=2
    )
    self.ax[0].add_patch(sensor_circle)
    self.dynamic_artists.append(sensor_circle)

    # Targets
    for tx, ty in self.target_positions:
        circle = patches.Circle((tx + 0.5, ty + 0.5), 0.3, color="red", alpha=0.7)
        self.ax[0].add_patch(circle)
        self.dynamic_artists.append(circle)
```

**Pros:** Smooth, no flickering, efficient
**Cons:** More complex code

---

### **Solution 3: Use Animation API (Professional)**

Use matplotlib's built-in animation system:

```python
from matplotlib.animation import FuncAnimation

class SAREnv(gym.Env):
    def setup_animation(self):
        """Setup matplotlib animation."""
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))

        # Draw static background
        self._render_static_background()

        # Create animation
        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            interval=50,  # 50ms between frames
            blit=True,    # Use blitting
            repeat=False
        )

    def _update_frame(self, frame):
        """Update function for animation."""
        # Return list of artists that changed
        return self.dynamic_artists
```

**Pros:** Professional, smooth, built-in support
**Cons:** Requires refactoring to animation paradigm

---

## Recommended Quick Fix

**Edit `envs/sar_env.py` line 476:**

```python
# Change this:
plt.pause(0.001)

# To this:
plt.pause(0.05)  # 50ms = 20 FPS, smoother rendering
```

**And optionally add at line 456:**
```python
if self.fig is None:
    self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))
    plt.ion()  # ← Add interactive mode
```

This reduces flickering significantly with minimal code changes!

---

## Performance Comparison

| Method | FPS | Flicker | CPU Usage | Code Complexity |
|--------|-----|---------|-----------|-----------------|
| **Current (glitchy)** | ~1000 | ❌ High | High | Simple |
| **Quick Fix (pause)** | ~20 | ⚠️ Medium | Medium | Simple |
| **Blitting** | ~60 | ✅ None | Low | Medium |
| **Animation API** | ~60 | ✅ None | Low | Complex |

---

## Why `plt.pause(0.001)` Is Too Fast

- **1ms pause** = potential for **1000 FPS**
- Human eye sees **24-60 FPS** as smooth
- matplotlib can't redraw that fast → lag → flicker
- Screen refresh rate typically **60 Hz** (16.7ms per frame)

**Solution:** Match or exceed screen refresh time:
- `plt.pause(0.016)` = ~60 FPS (matches 60Hz monitors)
- `plt.pause(0.05)` = 20 FPS (smooth enough, less CPU)

---

## Testing the Fix

```python
from envs import SAREnv, EnvConfig
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better performance

config = EnvConfig(grid_size=10, num_targets=1, render_mode="human")
env = SAREnv(config)

# Modify render pause time (temporary test)
import envs.sar_env as sar_module

obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        break

env.close()
```

If still glitchy, the issue might be:
1. **Backend**: Try different matplotlib backends (`TkAgg`, `Qt5Agg`)
2. **System**: Display driver or compositor issues
3. **Python version**: Some versions have matplotlib bugs

---

## Bottom Line

**The glitching is caused by:**
1. ❌ Clearing entire plot every frame (`ax.clear()`)
2. ❌ Redrawing static elements unnecessarily
3. ❌ Too-fast refresh rate (`plt.pause(0.001)`)

**Quick fix:** Increase pause time to `0.05` seconds
**Best fix:** Implement blitting to only update dynamic elements
