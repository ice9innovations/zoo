#!/bin/bash
# Extract Detectron2 segmentation masks from COCO images
# Usage: ./extract_masks.sh [input_dir] [output_dir] [confidence] [options]

set -e

# Parse command line arguments
INPUT_DIR=""
OUTPUT_DIR=""
CONFIDENCE="0.5"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --reset)
            EXTRA_ARGS="$EXTRA_ARGS --reset"
            shift
            ;;
        --no-resume)
            EXTRA_ARGS="$EXTRA_ARGS --no-resume"
            shift
            ;;
        --individual)
            EXTRA_ARGS="$EXTRA_ARGS --individual"
            shift
            ;;
        -t|--threshold)
            CONFIDENCE="$2"
            shift 2
            ;;
        -h|--help)
            echo "🎭 Detectron2 Mask Extraction"
            echo "Usage: $0 [input_dir] [output_dir] [options]"
            echo ""
            echo "Options:"
            echo "  -t, --threshold VALUE  Confidence threshold (default: 0.5)"
            echo "  --reset               Reset progress and start fresh"
            echo "  --no-resume           Disable resumption (start fresh)"
            echo "  --individual          Save individual object masks"
            echo "  -h, --help            Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use defaults"
            echo "  $0 /path/to/images /path/to/masks    # Custom paths"
            echo "  $0 --reset                           # Reset and restart"
            echo "  $0 --threshold 0.3                   # Lower confidence"
            echo "  $0 --individual                      # Save individual masks"
            exit 0
            ;;
        *)
            if [ -z "$INPUT_DIR" ]; then
                INPUT_DIR="$1"
            elif [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            else
                echo "❌ Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Set defaults if not provided
INPUT_DIR=${INPUT_DIR:-"/var/www/html/images/coco"}
OUTPUT_DIR=${OUTPUT_DIR:-"/home/sd/detectron2/masks"}

echo "🎭 Detectron2 Mask Extraction"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Confidence threshold: $CONFIDENCE"
echo "Options: $EXTRA_ARGS"
echo

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Input directory not found: $INPUT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check for existing progress in script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGRESS_FILE="$SCRIPT_DIR/detectron_masks_progress.json"
if [ -f "$PROGRESS_FILE" ] && [[ ! "$EXTRA_ARGS" =~ "--reset" ]] && [[ ! "$EXTRA_ARGS" =~ "--no-resume" ]]; then
    echo "📊 Found existing progress file - will resume from where we left off"
    echo "    Use --reset to start fresh or --no-resume to ignore progress"
fi

# Activate virtual environment and run the extraction
cd /home/sd/detectron2
echo "🚀 Starting mask extraction..."

# Check if virtual environment exists
if [ ! -d "detectron2_venv" ]; then
    echo "❌ Virtual environment not found: detectron2_venv"
    echo "Please create the virtual environment first"
    exit 1
fi

# Activate virtual environment and run
source detectron2_venv/bin/activate
python3 extract_detectron_masks.py \
    "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -t "$CONFIDENCE" \
    -p "*.jpg,*.jpeg,*.png,*.webp" \
    $EXTRA_ARGS

echo "✅ Mask extraction complete!"
echo "Masks saved to: $OUTPUT_DIR"

# Show final progress status
if [ -f "$PROGRESS_FILE" ]; then
    echo "📊 Progress file: $PROGRESS_FILE"
fi
