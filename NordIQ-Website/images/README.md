# NordIQ Images

This folder contains all images for the nordiqai.io website.

## Required Images

### High Priority (Needed for Launch)

1. **favicon.png** (32x32, 64x64)
   - NordIQ compass icon 🧭
   - Generate at: https://favicon.io/
   - Save as: `favicon.png` and `favicon.ico`

2. **logo.png** (Transparent PNG or SVG)
   - NordIQ wordmark + compass icon
   - Use for header navigation
   - Recommended size: 200x50px

3. **dashboard-preview.webp** (1920x1080 or larger)
   - Full screenshot of actual dashboard
   - Overview tab with fleet heatmap
   - Compress with WebP format
   - Used on homepage

4. **og-image.png** (1200x630)
   - For LinkedIn/Twitter social sharing
   - Include: Logo + tagline + key stat
   - Create at: https://www.canva.com/

### Medium Priority

5. **hero-background.webp** (2560x1440)
   - Nordic landscape or abstract tech pattern
   - Dark overlay for text readability
   - Background for hero sections

6. **screenshots/** (Dashboard tabs)
   - fleet-overview.webp
   - heatmap.webp
   - top-risks.webp
   - historical.webp
   - cost-avoidance.webp
   - alerting.webp

## Image Optimization

Convert to WebP for smaller file sizes:

```bash
# Install WebP tools
sudo apt-get install webp

# Convert PNG to WebP
cwebp dashboard.png -q 80 -o dashboard.webp

# Resize large images
convert large-image.png -resize 1920x1080 optimized.png
```

## Current Status

- [ ] favicon.png
- [ ] logo.png
- [ ] dashboard-preview.webp
- [ ] og-image.png
- [ ] hero-background.webp
- [ ] screenshots/ (6 images)

**Add images here and update this README when complete.**
