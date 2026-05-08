# Quick Fix: Adding Images to Journal Paper

## Current Issue
- **Journal Paper:** 0 images embedded
- **Final Report:** 12 images embedded and working perfectly
- **Solution:** Copy images from Final Report → Journal Paper

---

## Step-by-Step Fix (Python Script)

```python
#!/usr/bin/env python3
"""
Add images from Final Report to Journal Paper
Run this in: C:\Users\acer\Documents\projects\subgen_ai\docs\
"""

import zipfile
import shutil
from pathlib import Path

docs_dir = Path(".")
final_report = docs_dir / "SubGEN_AI_Final_Report_Updated.docx"
journal = docs_dir / "SubGEN_AI_Journal_Paper.docx"

# Backup journal before modifying
shutil.copy(journal, journal.with_stem(journal.stem + "_BACKUP"))
print(f"✅ Created backup: {journal.name}_BACKUP")

# Extract both files
with zipfile.ZipFile(final_report, 'r') as zip_ref:
    zip_ref.extractall("temp_report")

with zipfile.ZipFile(journal, 'r') as zip_ref:
    zip_ref.extractall("temp_journal")

# Copy media folder
final_report_media = Path("temp_report") / "word" / "media"
journal_media = Path("temp_journal") / "word" / "media"

if not journal_media.exists():
    journal_media.mkdir(parents=True)
    print(f"✅ Created {journal_media}")

for img in final_report_media.glob("*.png"):
    dest = journal_media / img.name
    shutil.copy(img, dest)
    print(f"   ✓ Copied {img.name}")

# Copy relationships file (contains image references)
final_report_rels = Path("temp_report") / "word" / "_rels" / "document.xml.rels"
journal_rels = Path("temp_journal") / "word" / "_rels" / "document.xml.rels"

shutil.copy(final_report_rels, journal_rels)
print(f"✅ Updated relationships file")

# Copy Content_Types.xml for image MIME types
final_report_ct = Path("temp_report") / "[Content_Types].xml"
journal_ct = Path("temp_journal") / "[Content_Types].xml"

shutil.copy(final_report_ct, journal_ct)
print(f"✅ Updated content types")

# Repackage journal DOCX
import os
os.remove(journal)

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, path)
            ziph.write(file_path, arcname)

with zipfile.ZipFile(journal, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipdir("temp_journal", zipf)

print(f"✅ Repacked {journal.name}")

# Cleanup
shutil.rmtree("temp_report")
shutil.rmtree("temp_journal")
print(f"✅ Cleaned up temporary files")

print("\n" + "=" * 60)
print("DONE! Journal Paper now has images from Final Report")
print("=" * 60)
```

---

## Manual Fix (Without Script)

If you prefer doing it manually via Word:

### Option A: Copy-Paste All Images

1. **Open Final Report** in Microsoft Word
2. Select all images (Ctrl+A, then copy images one by one)
3. **Open Journal Paper** in Microsoft Word
4. Paste each image at appropriate locations:
   - Image 1 (System Architecture) → After Section 3 introduction
   - Images 2-3 (Pipelines/Implementation) → Optional (not critical for journal)
5. **Save and close**

### Option B: Use Word's "Insert > Pictures" Feature

1. **Open Journal Paper** in Microsoft Word
2. Navigate to Section 3 "Proposed System Architecture"
3. Click Insert → Pictures → From this device
4. Select image1.png from Final Report's extracted word/media/ folder
5. Do the same for other key images (system architecture minimum required)

---

## What Images Should Go Where?

Based on Final Report structure, minimum images needed for journal:

| Image | Goes After | Reason |
|-------|-----------|--------|
| image1.png | Section 3 "Proposed System Architecture" | Required—Referenced as "Figure 1" |
| image5.png (QC diagram) | Section 4.2 | Optional but recommended |
| image10.png (WER chart) | Section 6.1 Results | Optional—Nice to visualize |

---

## Verification After Fix

After adding images, verify:

1. **Open Journal Paper** in Word or PDF reader
2. Check that images appear and are readable
3. Verify **No broken references** (red X marks)
4. Confirm **page layout** looks professional

---

## Expected Result

After fix:
```
Journal Paper
├── 0 images (before)
└── 12 images (after) ✅
```

File size will increase ~2-5 MB (due to embedded PNGs).

---

## Timing

- **Script method:** ~2 minutes to run
- **Manual method:** ~15-20 minutes
- **Total with verification:** 30 minutes maximum

---

## Support

If you run the Python script and get errors:

```
Error: FileNotFoundError: [Errno 2] No such file or directory
→ Make sure you run it from: C:\Users\acer\Documents\projects\subgen_ai\docs\
```

```
Error: Bad zipfile
→ Journal Paper may be corrupted. Use the _BACKUP file.
  Restore: Rename SubGEN_AI_Journal_Paper_BACKUP → SubGEN_AI_Journal_Paper.docx
  Try again.
```

---

**Priority:** 🔴 **CRITICAL** — Do this before faculty review  
**Time Required:** ⏱️ **30 minutes**  
**Difficulty:** ⭐ **Easy**

