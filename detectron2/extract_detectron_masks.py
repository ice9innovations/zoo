#!/usr/bin/env python3
"""
Extract Detectron2 instance segmentation masks from images and save as PNG files
Uses Mask R-CNN for comprehensive object segmentation (80 COCO classes)
"""

import os
import sys
import time
import argparse
import logging
import json
import signal
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from datetime import datetime

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProgressTracker:
    """Track progress and enable resumption of batch processing"""
    
    def __init__(self, progress_file):
        self.progress_file = Path(progress_file)
        self.data = {
            'started_at': None,
            'last_updated': None,
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'completed_files': [],
            'failed_file_list': [],
            'current_file': None,
            'stats': {
                'total_objects': 0,
                'total_pixels': 0,
                'avg_coverage': 0.0
            }
        }
        self._load_progress()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, saving progress...")
            self.save_progress()
            logger.info("Progress saved. Exiting gracefully.")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Kill signal
    
    def _load_progress(self):
        """Load existing progress if available"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    saved_data = json.load(f)
                    self.data.update(saved_data)
                logger.info(f"📊 Loaded progress: {self.data['processed_files']}/{self.data['total_files']} files completed")
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
    
    def save_progress(self):
        """Save current progress to file"""
        self.data['last_updated'] = datetime.now().isoformat()
        try:
            # Atomic write using temporary file
            temp_file = self.progress_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            temp_file.replace(self.progress_file)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def initialize_batch(self, total_files):
        """Initialize or resume batch processing"""
        if self.data['started_at'] is None:
            # New batch
            self.data['started_at'] = datetime.now().isoformat()
            self.data['total_files'] = total_files
            logger.info(f"🚀 Starting new batch: {total_files} files")
        else:
            # Resuming batch
            logger.info(f"🔄 Resuming batch from {self.data['processed_files']}/{total_files}")
        
        self.save_progress()
    
    def is_file_completed(self, file_path):
        """Check if file was already processed"""
        return str(file_path) in self.data['completed_files']
    
    def mark_file_started(self, file_path):
        """Mark file as currently being processed"""
        self.data['current_file'] = str(file_path)
        self.save_progress()
    
    def mark_file_completed(self, file_path, stats=None):
        """Mark file as successfully completed"""
        file_str = str(file_path)
        if file_str not in self.data['completed_files']:
            self.data['completed_files'].append(file_str)
            self.data['processed_files'] += 1
        
        if stats:
            self.data['stats']['total_objects'] += stats.get('num_objects', 0)
            self.data['stats']['total_pixels'] += stats.get('object_pixels', 0)
        
        self.data['current_file'] = None
        
        # Save progress every 10 files
        if self.data['processed_files'] % 10 == 0:
            self.save_progress()
    
    def mark_file_failed(self, file_path, error_msg):
        """Mark file as failed"""
        file_str = str(file_path)
        if file_str not in self.data['failed_file_list']:
            self.data['failed_file_list'].append({
                'file': file_str,
                'error': str(error_msg),
                'timestamp': datetime.now().isoformat()
            })
            self.data['failed_files'] += 1
        
        self.data['current_file'] = None
        self.save_progress()
    
    def get_remaining_files(self, all_files):
        """Get list of files that still need processing"""
        completed_set = set(self.data['completed_files'])
        return [f for f in all_files if str(f) not in completed_set]
    
    def print_summary(self):
        """Print final summary"""
        total = self.data['total_files']
        processed = self.data['processed_files']
        failed = self.data['failed_files']
        
        logger.info("="*50)
        logger.info("📊 BATCH PROCESSING SUMMARY")
        logger.info("="*50)
        logger.info(f"Total files: {total}")
        logger.info(f"Successfully processed: {processed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {(processed/total*100):.1f}%" if total > 0 else "N/A")
        
        if self.data['stats']['total_objects'] > 0:
            avg_objects = self.data['stats']['total_objects'] / processed if processed > 0 else 0
            logger.info(f"Total objects detected: {self.data['stats']['total_objects']}")
            logger.info(f"Average objects per image: {avg_objects:.1f}")
        
        if failed > 0:
            logger.info(f"\n❌ Failed files ({failed}):")
            for failure in self.data['failed_file_list'][-10:]:  # Show last 10 failures
                logger.info(f"  {Path(failure['file']).name}: {failure['error']}")
            if failed > 10:
                logger.info(f"  ... and {failed-10} more failures")
    
    def cleanup(self):
        """Clean up progress file after successful completion"""
        if self.progress_file.exists():
            self.progress_file.unlink()
            logger.info("✅ Progress file cleaned up")

class Detectron2SegmentationExtractor:
    def __init__(self, confidence_threshold=0.5):
        """Initialize Detectron2 Mask R-CNN model for segmentation"""
        self.confidence_threshold = confidence_threshold
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU is required for Detectron2.")
        
        # Setup configuration
        cfg = get_cfg()
        
        # Use Mask R-CNN config (supports instance segmentation)
        cfg.merge_from_file("/home/sd/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        # Set model weights for Mask R-CNN
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        
        # Set confidence threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        
        # Performance settings
        cfg.TEST.DETECTIONS_PER_IMAGE = 20
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        
        # Use GPU
        cfg.MODEL.DEVICE = "cuda"
        
        # Initialize predictor
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        
        logger.info("✅ Detectron2 Mask R-CNN initialized for segmentation")
        logger.info(f"Confidence threshold: {confidence_threshold}")
    
    def extract_masks(self, image_path, output_dir, save_individual=False):
        """
        Extract segmentation masks from image
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save mask PNGs
            save_individual: If True, save separate mask for each detected object
        
        Returns:
            dict: Mask statistics and object information
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            height, width = image.shape[:2]
            
            # Run Detectron2 inference
            outputs = self.predictor(image)
            
            # Extract predictions
            instances = outputs["instances"].to("cpu")
            
            if len(instances) == 0:
                logger.info(f"No objects detected in {image_path.name}")
                return None
            
            # Get masks, classes, and scores
            masks = instances.pred_masks.numpy()  # Shape: (N, H, W)
            classes = instances.pred_classes.numpy()
            scores = instances.scores.numpy()
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Base filename (without extension)
            base_filename = image_path.stem
            
            # Create combined mask (all objects)
            combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            combined_mask_path = output_path / f"{base_filename}_mask.png"
            
            # Save combined mask
            mask_image = Image.fromarray(combined_mask, mode='L')
            mask_image.save(combined_mask_path)
            
            # Optionally save individual object masks
            individual_masks = []
            if save_individual:
                for i, (mask, cls, score) in enumerate(zip(masks, classes, scores)):
                    class_name = self.metadata.thing_classes[cls]
                    individual_mask = (mask.astype(np.uint8) * 255)
                    individual_path = output_path / f"{base_filename}_{class_name}_{i:02d}_mask.png"
                    
                    individual_image = Image.fromarray(individual_mask, mode='L')
                    individual_image.save(individual_path)
                    
                    individual_masks.append({
                        'class': class_name,
                        'confidence': float(score),
                        'mask_file': str(individual_path),
                        'pixels': int(np.sum(mask))
                    })
            
            # Calculate statistics
            total_pixels = height * width
            object_pixels = int(np.sum(combined_mask > 0))
            coverage_ratio = object_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # Object summary
            detected_objects = []
            for cls, score in zip(classes, scores):
                class_name = self.metadata.thing_classes[cls]
                detected_objects.append({
                    'class': class_name,
                    'confidence': float(score)
                })
            
            stats = {
                'num_objects': len(instances),
                'object_pixels': object_pixels,
                'total_pixels': total_pixels,
                'coverage_ratio': round(coverage_ratio, 4),
                'detected_objects': detected_objects,
                'individual_masks': individual_masks if save_individual else [],
                'combined_mask_file': str(combined_mask_path),
                'image_dimensions': {'height': height, 'width': width},
                'confidence_threshold': self.confidence_threshold
            }
            
            # Log summary
            obj_summary = ", ".join([f"{obj['class']}({obj['confidence']:.2f})" for obj in detected_objects])
            logger.info(f"✅ {image_path.name}: {len(instances)} objects ({coverage_ratio:.1%} coverage) - {obj_summary}")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to extract masks from {image_path}: {e}")
            return None
    
    def process_directory(self, input_dir, output_dir, file_pattern="*.jpg", save_individual=False, resume=True):
        """
        Process all images in a directory with progress tracking and resumption
        
        Args:
            input_dir: Directory containing source images
            output_dir: Directory to save mask PNGs
            file_pattern: Glob pattern for image files
            save_individual: Save individual object masks
            resume: Enable resumption from previous runs
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress tracker in script directory, not output directory
        script_dir = Path(__file__).parent
        progress_file = script_dir / "detectron_masks_progress.json"
        tracker = ProgressTracker(progress_file) if resume else None
        
        # Find all images
        image_files = list(input_path.glob(file_pattern))
        if not image_files:
            logger.warning(f"No files found matching {file_pattern} in {input_dir}")
            return
        
        # Sort for consistent processing order
        image_files.sort()
        
        if tracker:
            tracker.initialize_batch(len(image_files))
            
            # Get remaining files to process
            remaining_files = tracker.get_remaining_files(image_files)
            if len(remaining_files) < len(image_files):
                logger.info(f"Resuming: {len(remaining_files)} files remaining out of {len(image_files)}")
            image_files = remaining_files
        else:
            logger.info(f"Found {len(image_files)} images to process")
        
        if not image_files:
            logger.info("✅ All files already processed!")
            if tracker:
                tracker.print_summary()
            return
        
        # Process each image
        start_time = time.time()
        
        try:
            for i, image_file in enumerate(image_files, 1):
                if tracker:
                    tracker.mark_file_started(image_file)
                
                # Progress logging
                if i % 10 == 0 or i == len(image_files):
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta_seconds = (len(image_files) - i) / rate if rate > 0 else 0
                    eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                    logger.info(f"Progress: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%) "
                              f"Rate: {rate:.1f}/s ETA: {eta_str}")
                
                try:
                    # Extract masks
                    stats = self.extract_masks(image_file, output_path, save_individual)
                    
                    if stats:
                        if tracker:
                            tracker.mark_file_completed(image_file, stats)
                        logger.debug(f"✅ {image_file.name}: {stats['num_objects']} objects")
                    else:
                        error_msg = "No objects detected or processing failed"
                        if tracker:
                            tracker.mark_file_failed(image_file, error_msg)
                        logger.warning(f"⚠️  {image_file.name}: {error_msg}")
                
                except Exception as e:
                    error_msg = f"Processing error: {str(e)}"
                    if tracker:
                        tracker.mark_file_failed(image_file, error_msg)
                    logger.error(f"❌ {image_file.name}: {error_msg}")
                
                # Memory management - force garbage collection every 50 files
                if i % 50 == 0:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user. Progress has been saved.")
            if tracker:
                tracker.save_progress()
            return
        
        # Final summary
        elapsed = time.time() - start_time
        if tracker:
            tracker.save_progress()
            tracker.print_summary()
            
            # Clean up progress file if everything completed successfully
            if tracker.data['processed_files'] + tracker.data['failed_files'] >= tracker.data['total_files']:
                tracker.cleanup()
        else:
            logger.info(f"Completed processing {len(image_files)} files in {elapsed:.1f}s")

def main():
    parser = argparse.ArgumentParser(description='Extract Detectron2 instance segmentation masks')
    parser.add_argument('input', help='Input directory or single image file')
    parser.add_argument('-o', '--output', required=True, help='Output directory for masks')
    parser.add_argument('-p', '--pattern', default='*.{jpg,jpeg,png,webp}', 
                       help='File pattern for batch processing')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--individual', action='store_true',
                       help='Save individual object masks in addition to combined mask')
    parser.add_argument('--single', action='store_true',
                       help='Process single file instead of directory')
    parser.add_argument('--no-resume', action='store_true',
                       help='Disable resumption (start fresh)')
    parser.add_argument('--reset', action='store_true',
                       help='Reset progress and start fresh')
    
    args = parser.parse_args()
    
    # Handle reset option
    if args.reset:
        output_dir = Path(args.output)
        progress_file = output_dir / "detectron_masks_progress.json"
        if progress_file.exists():
            progress_file.unlink()
            logger.info("🔄 Progress file reset - starting fresh")
    
    # Initialize extractor
    extractor = Detectron2SegmentationExtractor(confidence_threshold=args.threshold)
    
    if args.single or Path(args.input).is_file():
        # Single file processing
        input_file = Path(args.input)
        output_dir = Path(args.output)
        
        stats = extractor.extract_masks(input_file, output_dir, args.individual)
        if stats:
            print(f"Combined mask saved to: {stats['combined_mask_file']}")
            print(f"Objects detected: {stats['num_objects']}")
            print(f"Coverage: {stats['coverage_ratio']:.1%}")
            for obj in stats['detected_objects']:
                print(f"  - {obj['class']}: {obj['confidence']:.2f}")
        else:
            print("Failed to extract masks")
            sys.exit(1)
    else:
        # Directory processing
        # Handle multiple extensions
        patterns = args.pattern.split(',') if ',' in args.pattern else [args.pattern]
        
        for pattern in patterns:
            pattern = pattern.strip()
            if pattern.startswith('*.{') and pattern.endswith('}'):
                # Handle *.{jpg,jpeg,png} format
                extensions = pattern[3:-1].split(',')
                for ext in extensions:
                    ext_pattern = f"*.{ext.strip()}"
                    extractor.process_directory(args.input, args.output, ext_pattern, args.individual, resume=not args.no_resume)
            else:
                extractor.process_directory(args.input, args.output, pattern, args.individual, resume=not args.no_resume)

if __name__ == '__main__':
    main()