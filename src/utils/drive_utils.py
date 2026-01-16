"""
Google Drive integration utilities for Google Colab.

This module provides functions to mount Google Drive, manage paths,
and sync data between Colab and Drive for the fall detection project.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union
import shutil


def mount_drive(mount_point: str = "/content/drive") -> Path:
    """
    Mount Google Drive in Google Colab.

    Args:
        mount_point: Path where Google Drive will be mounted

    Returns:
        Path object pointing to the mounted drive

    Example:
        >>> drive_root = mount_drive()
        >>> print(f"Drive mounted at: {drive_root}")
    """
    try:
        from google.colab import drive
        drive.mount(mount_point)
        print(f"✓ Google Drive mounted at: {mount_point}")
        return Path(mount_point)
    except ImportError:
        print("⚠ Warning: Not running in Google Colab. Skipping drive mount.")
        return Path(mount_point)
    except Exception as e:
        print(f"✗ Error mounting drive: {e}")
        raise


def setup_paths(
    project_name: str = "safeguard-vision-ai",
    drive_root: Union[str, Path] = "/content/drive",
    create_dirs: bool = True
) -> Dict[str, Path]:
    """
    Setup project paths in Google Drive for consistent access.

    Args:
        project_name: Name of the project folder in Drive
        drive_root: Root path of mounted Google Drive
        create_dirs: Whether to create directories if they don't exist

    Returns:
        Dictionary with project paths:
            - 'project_root': Main project folder
            - 'data_raw': Raw video data
            - 'data_processed': Processed keypoints
            - 'data_splits': Train/val/test splits
            - 'checkpoints': Model checkpoints
            - 'logs': Training logs
            - 'figures': Result visualizations

    Example:
        >>> paths = setup_paths()
        >>> print(f"Raw data at: {paths['data_raw']}")
    """
    drive_root = Path(drive_root)
    project_root = drive_root / "MyDrive" / project_name

    paths = {
        'project_root': project_root,
        'data_raw': project_root / 'data' / 'raw',
        'data_processed': project_root / 'data' / 'processed',
        'data_splits': project_root / 'data' / 'splits',
        'checkpoints': project_root / 'results' / 'checkpoints',
        'logs': project_root / 'results' / 'logs',
        'figures': project_root / 'results' / 'figures',
    }

    if create_dirs:
        for name, path in paths.items():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created/verified: {name} -> {path}")

    return paths


def sync_data(
    source: Union[str, Path],
    destination: Union[str, Path],
    pattern: str = "*",
    verbose: bool = True
) -> int:
    """
    Sync data between local Colab storage and Google Drive.

    Useful for faster processing: copy from Drive to Colab /content,
    process data quickly, then sync results back to Drive.

    Args:
        source: Source directory path
        destination: Destination directory path
        pattern: Glob pattern for files to sync (default: all files)
        verbose: Print progress information

    Returns:
        Number of files synced

    Example:
        >>> # Copy processed data from Colab to Drive
        >>> sync_data('/content/processed', paths['data_processed'])
    """
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        raise ValueError(f"Source path does not exist: {source}")

    destination.mkdir(parents=True, exist_ok=True)

    files = list(source.rglob(pattern))
    count = 0

    for file_path in files:
        if file_path.is_file():
            relative_path = file_path.relative_to(source)
            dest_path = destination / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(file_path, dest_path)
            count += 1

            if verbose:
                print(f"✓ Synced: {relative_path}")

    if verbose:
        print(f"\n✓ Total files synced: {count}")

    return count


def save_checkpoint(
    checkpoint_dict: Dict,
    save_path: Union[str, Path],
    drive_path: Optional[Union[str, Path]] = None,
    backup: bool = True
) -> None:
    """
    Save model checkpoint locally and optionally to Google Drive.

    Args:
        checkpoint_dict: Dictionary containing model state, optimizer state, etc.
        save_path: Local path to save checkpoint
        drive_path: Optional Google Drive path for backup
        backup: If True and drive_path provided, save copy to Drive

    Example:
        >>> checkpoint = {
        ...     'epoch': 10,
        ...     'model_state_dict': model.state_dict(),
        ...     'optimizer_state_dict': optimizer.state_dict(),
        ...     'loss': 0.25,
        ... }
        >>> save_checkpoint(checkpoint, 'model.pth', paths['checkpoints'])
    """
    import torch

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save locally
    torch.save(checkpoint_dict, save_path)
    print(f"✓ Checkpoint saved locally: {save_path}")

    # Backup to Drive if requested
    if backup and drive_path:
        drive_path = Path(drive_path)
        drive_path.mkdir(parents=True, exist_ok=True)

        drive_file = drive_path / save_path.name
        shutil.copy2(save_path, drive_file)
        print(f"✓ Checkpoint backed up to Drive: {drive_file}")


def get_dataset_info(data_path: Union[str, Path]) -> Dict[str, int]:
    """
    Get information about dataset in a directory.

    Args:
        data_path: Path to dataset directory

    Returns:
        Dictionary with dataset statistics

    Example:
        >>> info = get_dataset_info(paths['data_raw'])
        >>> print(f"Total videos: {info['total_files']}")
    """
    data_path = Path(data_path)

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = [
        f for f in data_path.rglob('*')
        if f.suffix.lower() in video_extensions
    ]

    return {
        'total_files': len(video_files),
        'total_size_mb': sum(f.stat().st_size for f in video_files) / (1024 * 1024),
        'extensions': list(set(f.suffix for f in video_files)),
    }


def clone_repo(
    repo_url: str = "https://github.com/hugoangeles0810/safeguard-vision-ai.git",
    target_dir: str = "/content/safeguard-vision-ai"
) -> Path:
    """
    Clone the project repository in Google Colab.

    Args:
        repo_url: GitHub repository URL
        target_dir: Directory where to clone the repo

    Returns:
        Path to the cloned repository

    Example:
        >>> repo_path = clone_repo()
        >>> os.chdir(repo_path)
    """
    import subprocess

    target_path = Path(target_dir)

    if target_path.exists():
        print(f"⚠ Repository already exists at: {target_path}")
        return target_path

    print(f"Cloning repository from {repo_url}...")
    subprocess.run(['git', 'clone', repo_url, str(target_path)], check=True)
    print(f"✓ Repository cloned to: {target_path}")

    return target_path
