import tensorflow as tf
import matplotlib.pyplot as plt
import json
from model.training import (
    parse_filenames_and_bboxes_from_json,
    parse_image_and_encode_bboxes,
    create_dataset_detection,
    preprocessing_layers_detection,
)

def draw_boxes(image, boxes, labels=None, color=(255, 0, 0)):
    """Draw bounding boxes on the image."""
    image = image.numpy() if isinstance(image, tf.Tensor) else image
    image = image.copy()
    height, width = image.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    for i, box in enumerate(boxes):
        y_min, x_min, y_max, x_max = box
        y_min, y_max = int(y_min * height), int(y_max * height)
        x_min, x_max = int(x_min * width), int(x_max * width)
        
        # Draw rectangle
        image[y_min:y_min+2, x_min:x_max] = color  # Top
        image[y_max-2:y_max, x_min:x_max] = color  # Bottom
        image[y_min:y_max, x_min:x_min+2] = color  # Left
        image[y_min:y_max, x_max-2:x_max] = color  # Right
        
        # Draw label if provided
        if labels is not None and i < len(labels):
            label = labels[i]
            # Use simple text instead of cv2
            plt.text(x_min, y_min - 5, label, color=tuple(c/255 for c in color), 
                    bbox=dict(facecolor='white', alpha=0.7))
    
    return image

def visualize_resizing_pipeline(json_path, target_size=(384, 384)):
    """Visualize the effect of resizing on images and their bounding boxes."""
    # Get first image and its annotations
    image_filenames, bbox_labels, bbox_coords = parse_filenames_and_bboxes_from_json(
        json_path,
        all_labels=["pizza"]
    )
    
    # Create the dataset using training pipeline
    train_dataset, _, _ = create_dataset_detection(
        filenames=image_filenames,
        classes=bbox_labels,
        boxes=bbox_coords,
        all_labels=["pizza"],
        src_bbox_format="rel_yxyx",
        tgt_bbox_format="rel_yxyx",
        target_shape=target_size,
        train_split=0.8,
        batch_size=1  # Set to 1 to easily get first image
    )
    
    # Get first image and boxes from dataset
    first_batch = next(iter(train_dataset))
    resized_image = first_batch[0][0]  # First image from first batch
    resized_boxes = first_batch[1]['boxes'][0]  # First set of boxes from first batch
    
    # Load original image directly
    original_image = tf.io.read_file(image_filenames[0])
    original_image = tf.io.decode_image(original_image, channels=3)
    original_boxes = tf.constant(bbox_coords[0])
    
    # Cast images to uint8
    original_image = tf.cast(original_image, tf.uint8)
    resized_image = tf.cast(resized_image, tf.uint8)
    
    # Draw boxes on both images
    original_with_boxes = draw_boxes(
        original_image,
        original_boxes,
        labels=None,
        color=(0, 255, 0)  # Green for original
    )
    
    resized_with_boxes = draw_boxes(
        resized_image,
        resized_boxes,
        labels=None,
        color=(255, 0, 0)  # Red for resized
    )
    
    # Plot results
    plt.figure(1)
    plt.title(f"Original Image\n{original_image.shape}")
    plt.imshow(original_with_boxes)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    
    plt.figure(2)
    plt.title(f"Resized Image (After Dataset Pipeline)\n{resized_image.shape}")
    plt.imshow(resized_with_boxes)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    
    plt.show()

def visualize_resizing_parsed(json_path, target_size=(384, 384)):
    """Visualize the effect of resizing on images and their bounding boxes."""
    # Get first image and its annotations using training.py function
    image_filenames, bbox_labels, bbox_coords = parse_filenames_and_bboxes_from_json(
        json_path,
        all_labels=["pizza"]  # Pass None to get all labels from the dataset
    )
    
    # Load original image directly without any resizing
    original_image = tf.io.read_file(image_filenames[0])
    original_image = tf.io.decode_image(original_image, channels=3)
    
    # Prepare data for resized version
    first_image_data = {
        "images": image_filenames[0],
        "bounding_boxes": {
            "boxes": tf.ragged.constant([bbox_coords[0]], ragged_rank=1),
            "classes": tf.ragged.constant([bbox_labels[0]])
        }
    }
    
    # Get resized version through training pipeline
    original_data = parse_image_and_encode_bboxes(
        first_image_data,
        all_labels=["pizza"],
        src_bbox_format="rel_yxyx",
        tgt_bbox_format="rel_yxyx",
        img_size=target_size
    )
    
    # Cast images to uint8
    original_image = tf.cast(original_image, tf.uint8)
    resized_image = tf.cast(original_data["images"], tf.uint8)
    
    # Get original and resized images with their boxes
    original_boxes = bbox_coords[0]
    resized_boxes = original_data["bounding_boxes"]["boxes"]
    
    # Draw boxes on both images without labels
    original_with_boxes = draw_boxes(
        original_image,
        original_boxes,
        labels=None,  # Remove labels
        color=(0, 255, 0)  # Green for original
    )
    
    resized_with_boxes = draw_boxes(
        resized_image,
        resized_boxes.to_tensor()[0],
        labels=None,  # Remove labels
        color=(255, 0, 0)  # Red for resized
    )
    
    # Plot results - remove figsize to let images determine the window size
    plt.figure()
    
    # Create two separate figures instead of subplots to maintain true sizes
    plt.figure(1)
    plt.title(f"Original Image\n{original_image.shape}")
    plt.imshow(original_with_boxes)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    
    plt.figure(2)
    plt.title(f"Resized Image (Preserved Aspect Ratio)\n{resized_image.shape}")
    plt.imshow(resized_with_boxes)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    
    # Remove tight_layout as we're using separate figures
    plt.show()
    
    # Print original and transformed coordinates
    print("\nBounding Box Coordinates (y_min, x_min, y_max, x_max):")
    print("\nOriginal:")
    for i, (box, label) in enumerate(zip(original_boxes, bbox_labels[0])):
        print(f"{label}: {box}")
    
    print("\nAfter Resizing:")
    for i, (box, label) in enumerate(zip(resized_boxes, bbox_labels[0])):
        print(f"{label}: {box.numpy()}")

def test_preprocessing_layers(json_path, target_size=(384, 384)):
    """Test if preprocessing layers are functioning correctly by visualizing their output.
    
    Args:
        json_path: Path to the dataset JSON file
        target_size: Tuple of (height, width) for target image size
    """
    # Get first image and its annotations
    image_filenames, bbox_labels, bbox_coords = parse_filenames_and_bboxes_from_json(
        json_path,
        all_labels=["pizza"]
    )
    
    # Load original image
    original_image = tf.io.read_file(image_filenames[0])
    original_image = tf.io.decode_image(original_image, channels=3)
    original_image = tf.cast(original_image, tf.uint8)
    
    # Draw boxes on original image
    original_with_boxes = draw_boxes(
        original_image,
        bbox_coords[0],
        labels=bbox_labels[0],
        color=(0, 255, 0)  # Green for original
    )
    
    # Create preprocessing layers
    preprocessing = preprocessing_layers_detection(
        target_shape=(target_size[0], target_size[1], 3)
    )
    
    # Apply preprocessing
    # Add batch dimension for preprocessing
    batched_image = tf.expand_dims(original_image, 0)
    processed_image = preprocessing(batched_image)
    # Remove batch dimension for display
    processed_image = tf.squeeze(processed_image)
    processed_image = tf.cast(processed_image, tf.uint8)
    
    # Calculate scaling factors for bounding boxes
    orig_height, orig_width = original_image.shape[:2]
    new_height, new_width = processed_image.shape[:2]
    
    # Adjust bounding boxes for the new size
    processed_boxes = []
    for box in bbox_coords[0]:
        y_min, x_min, y_max, x_max = box
        # Scale coordinates based on the new dimensions
        new_y_min = y_min * (new_height / orig_height)
        new_x_min = x_min * (new_width / orig_width)
        new_y_max = y_max * (new_height / orig_height)
        new_x_max = x_max * (new_width / orig_width)
        processed_boxes.append([new_y_min, new_x_min, new_y_max, new_x_max])

    # print original and processed boxes
    print("Original boxes:")
    print(bbox_coords[0])
    print("Processed boxes:")
    print(processed_boxes)
    
    # Draw boxes on processed image
    processed_with_boxes = draw_boxes(
        processed_image,
        processed_boxes,
        labels=bbox_labels[0],
        color=(255, 0, 0)  # Red for processed
    )
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original Image with Boxes\n{original_image.shape}")
    plt.imshow(original_with_boxes)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"After Preprocessing with Boxes\n{processed_image.shape}")
    plt.imshow(processed_with_boxes)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print shape information and bounding box coordinates
    print("\nImage Shapes:")
    print(f"Original: {original_image.shape}")
    print(f"Processed: {processed_image.shape}")
    print(f"\nTarget shape was: {target_size}")
    
    print("\nBounding Box Coordinates (y_min, x_min, y_max, x_max):")
    print("\nOriginal:")
    for i, (box, label) in enumerate(zip(bbox_coords[0], bbox_labels[0])):
        print(f"{label}: {box}")
    
    print("\nAfter Preprocessing:")
    for i, (box, label) in enumerate(zip(processed_boxes, bbox_labels[0])):
        print(f"{label}: {box}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True,
                       help="Path to the dataset JSON file containing ABSOLUTE paths")
    parser.add_argument("--target_height", type=int, default=384,
                       help="Target height for resizing")
    parser.add_argument("--target_width", type=int, default=384,
                       help="Target width for resizing")
    args = parser.parse_args()
    visualize_resizing_pipeline(
        args.dataset_file,
        target_size=(args.target_height, args.target_width)
    )
    visualize_resizing_parsed(
        args.dataset_file,
        target_size=(args.target_height, args.target_width)
    )
    test_preprocessing_layers(
        args.dataset_file,
        target_size=(args.target_height, args.target_width)
    ) 