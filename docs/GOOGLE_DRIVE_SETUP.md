# Google Drive Integration Guide

This guide explains how to use Google Drive with Google Colab for the SafeGuard Vision AI project.

## Overview

The project uses Google Drive for:
- **Dataset storage**: Raw videos and processed keypoints
- **Model checkpoints**: Saving trained models
- **Results**: Training logs and visualizations
- **Persistence**: Data persists across Colab sessions

## Setup Instructions

### 1. Organize Your Google Drive

Create the following folder structure in your Google Drive:

```
Google Drive/
└── MyDrive/
    └── safeguard-vision-ai/
        ├── data/
        │   ├── raw/
        │   │   ├── ur_fall/        # UR Fall Detection videos
        │   │   └── le2i/           # Le2i Fall Detection videos
        │   ├── processed/          # Will store extracted keypoints
        │   └── splits/             # Train/val/test splits
        └── results/
            ├── checkpoints/        # Model checkpoints
            ├── logs/               # Training logs
            └── figures/            # Visualizations
```

### 2. Upload Datasets

1. Download datasets from:
   - [UR Fall Detection Dataset](http://fenix.ur.edu.pl/~mkepski/ds/uf.html)
   - [Le2i Fall Detection Dataset](https://imvia.u-bourgogne.fr/en/database/fall-detection-dataset-2.html)

2. Upload videos to the corresponding folders in Google Drive:
   - `MyDrive/safeguard-vision-ai/data/raw/ur_fall/`
   - `MyDrive/safeguard-vision-ai/data/raw/le2i/`

### 3. Use Notebooks in Google Colab

All notebooks automatically handle Drive integration:

1. **Open a notebook in Colab:**
   - Upload the notebook to Google Colab, or
   - Open it directly from GitHub in Colab

2. **Run the setup cells:**
   ```python
   # First cell - Check environment
   try:
       import google.colab
       IN_COLAB = True
   except ImportError:
       IN_COLAB = False
   ```

3. **Mount Drive and setup paths:**
   ```python
   # Second cell - Mount Drive
   from google.colab import drive
   drive.mount('/content/drive')

   # Setup project paths
   from src.utils.drive_utils import setup_paths
   paths = setup_paths(project_name='safeguard-vision-ai')
   ```

4. **Access your data:**
   ```python
   # Paths are now available
   print(f"Raw data: {paths['data_raw']}")
   print(f"Processed: {paths['data_processed']}")
   print(f"Checkpoints: {paths['checkpoints']}")
   ```

## Using Drive Utilities

### Import the utilities

```python
from src.utils.drive_utils import (
    mount_drive,
    setup_paths,
    sync_data,
    save_checkpoint,
    get_dataset_info,
    clone_repo
)
```

### Mount Google Drive

```python
drive_root = mount_drive()
# Output: ✓ Google Drive mounted at: /content/drive
```

### Setup project paths

```python
paths = setup_paths(
    project_name='safeguard-vision-ai',
    create_dirs=True  # Create directories if they don't exist
)
```

Returns a dictionary with all project paths:
- `paths['project_root']`
- `paths['data_raw']`
- `paths['data_processed']`
- `paths['data_splits']`
- `paths['checkpoints']`
- `paths['logs']`
- `paths['figures']`

### Sync data between Colab and Drive

For faster processing, you can work in Colab's local storage and sync to Drive:

```python
# Copy data from Drive to Colab for faster processing
sync_data(
    source=paths['data_raw'],
    destination='/content/data_local',
    pattern='*.mp4'
)

# Process data locally...

# Sync results back to Drive
sync_data(
    source='/content/processed',
    destination=paths['data_processed']
)
```

### Save model checkpoints

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'metrics': metrics
}

# Save locally and backup to Drive
save_checkpoint(
    checkpoint_dict=checkpoint,
    save_path='/content/model.pth',
    drive_path=paths['checkpoints'],
    backup=True
)
```

### Get dataset information

```python
info = get_dataset_info(paths['data_raw'])
print(f"Total videos: {info['total_files']}")
print(f"Total size: {info['total_size_mb']:.2f} MB")
print(f"Extensions: {info['extensions']}")
```

### Clone repository in Colab

```python
repo_path = clone_repo(
    repo_url="https://github.com/hugoangeles0810/safeguard-vision-ai.git",
    target_dir="/content/safeguard-vision-ai"
)
```

## Workflow Example

### Training a model in Colab

```python
# 1. Setup
from google.colab import drive
drive.mount('/content/drive')

from src.utils.drive_utils import setup_paths, save_checkpoint
paths = setup_paths()

# 2. Load data from Drive
train_loader = create_dataloader(paths['data_processed'])

# 3. Train model
for epoch in range(num_epochs):
    # Training loop...

    # Save checkpoint to Drive every N epochs
    if epoch % 5 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        save_checkpoint(
            checkpoint,
            f'/content/checkpoint_epoch_{epoch}.pth',
            paths['checkpoints'],
            backup=True
        )

# 4. Save final model to Drive
save_checkpoint(
    final_checkpoint,
    '/content/best_model.pth',
    paths['checkpoints'],
    backup=True
)
```

## Best Practices

### 1. Data Organization
- Keep raw videos in Drive (large files)
- Store processed keypoints in Drive (smaller, reusable)
- Back up important checkpoints to Drive

### 2. Performance Optimization
- For intensive processing, copy data to Colab's local storage (`/content/`)
- Process locally for speed
- Sync results back to Drive when done

### 3. Checkpoint Management
- Save checkpoints regularly during training
- Always backup best models to Drive
- Use descriptive filenames with timestamps or metrics

### 4. Collaborative Work
- Share the Drive folder with team members
- Use consistent folder structure
- Document any changes to data organization

## Troubleshooting

### Drive not mounting

```python
# Manually mount with force_remount
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Permission denied errors

- Ensure you have write permissions to the Drive folder
- Check that the folder path is correct
- Try creating directories manually first

### Out of space in Colab

```python
# Clean up local storage
!rm -rf /content/temp_data
!rm /content/*.pth

# Work directly from Drive (slower but no space issues)
```

### Slow data access

```python
# Copy frequently accessed data to Colab local storage
import shutil
shutil.copytree(paths['data_processed'], '/content/data_local')
# Use /content/data_local for training
```

## Additional Resources

- [Google Colab Documentation](https://colab.research.google.com/)
- [Google Drive API](https://developers.google.com/drive)
- [Project Repository](https://github.com/hugoangeles0810/safeguard-vision-ai)

## Support

If you encounter issues with Google Drive integration:
1. Check this documentation
2. Review the code in `src/utils/drive_utils.py`
3. Open an issue on GitHub with error details
