# CLIP Label Management

The CLIP service loads classification labels from text files in the `./labels/` folder, making it easy to add, remove, or modify the objects it can recognize.

## Label Files Location

All label files are stored in `./CLIP/labels/` for better organization.

## Active Label Files

- **`labels_coco.txt`** - Standard COCO dataset objects (80 classes)
- **`labels_extra.txt`** - Additional useful labels (diagram, chart, etc.)
- **`labels_objectnet.txt`** - Extended object recognition from ObjectNet

## Disabled Label Files

- **`labels_flowers.txt`** - Flower types (disabled - too specialized for general use)

## Adding New Labels

1. **Add to existing file**: Edit any `.txt` file in `./labels/` and add new labels (one per line)
2. **Create new category**: Create a new `.txt` file in `./labels/` and add it to `LABEL_FILES` in `REST.py`
3. **Enable/disable categories**: Comment/uncomment files in the `LABEL_FILES` array
4. **Restart service**: Changes take effect after service restart

## Label File Format

```
# Comments start with #
person
bicycle
car
# Empty lines are ignored

hot dog
cell phone
```

## Alternative Label Categories

Instead of flowers, consider these more varied label sets:

### Activities & Actions
`running, cooking, reading, dancing, working, driving, etc.`

### Concepts & Abstractions  
`art, science, landscape, portrait, vintage, modern, colorful, etc.`

### Digital & Web Content
`screenshot, website, app, code, logo, interface, social media, etc.`

## Current Label Count

The service loads approximately 400+ unique labels total:
- COCO: ~80 standard object detection classes
- Extra: ~4 additional useful categories  
- ObjectNet: ~300+ household and everyday objects

## Tips

- Use spaces in multi-word labels (they'll be normalized to underscores for emoji lookup)
- Avoid duplicates (they're automatically filtered)
- Keep labels descriptive but concise
- Test new labels with the health endpoint to verify count
- Store all files in `./labels/` folder for organization

## Emoji Mapping

Labels with spaces (e.g., "hot dog") are automatically normalized to underscores ("hot_dog") when looking up emojis in `emojis.json`.