import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageDraw
import warnings

warnings.filterwarnings("ignore")

print("🔄 Initializing CLIP Dispute Resolution Engine...")

# 1. Load Pre-trained CLIP Model
model_id = "openai/clip-vit-base-patch32"
print("⏳ Loading CLIP model...")
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
print("✅ CLIP Model Loaded.")

# 2. Create local "Before" and "After" images (no internet needed!)
print("🎨 Creating local test images...")

# "Before" image: A messy red workspace
image_before = Image.new('RGB', (224, 224), color='darkred')
draw_before = ImageDraw.Draw(image_before)
draw_before.rectangle([20, 20, 100, 100], fill='brown')
draw_before.rectangle([120, 50, 200, 180], fill='orange')

# "After" image: A clean green workspace (very different visually)
image_after = Image.new('RGB', (224, 224), color='lightgreen')
draw_after = ImageDraw.Draw(image_after)
draw_after.ellipse([50, 50, 180, 180], fill='white')
draw_after.line([10, 10, 210, 210], fill='blue', width=5)

print("✅ Test images created.")

# 3. Process Images through CLIP's VISION model only
print("🔄 Extracting visual features...")
inputs_before = processor(images=image_before, return_tensors="pt")
inputs_after = processor(images=image_after, return_tensors="pt")

with torch.no_grad():
    # Get the full output object
    output_before = model.get_image_features(pixel_values=inputs_before['pixel_values'])
    output_after = model.get_image_features(pixel_values=inputs_after['pixel_values'])
    
    # Extract the actual tensor from the output object
    # Try pooler_output first, fallback to last_hidden_state
    if hasattr(output_before, 'pooler_output'):
        features_before = output_before.pooler_output
        features_after = output_after.pooler_output
    elif hasattr(output_before, 'last_hidden_state'):
        features_before = output_before.last_hidden_state[:, 0, :]  # Take [CLS] token
        features_after = output_after.last_hidden_state[:, 0, :]
    else:
        # If it's already a tensor, use it directly
        features_before = output_before
        features_after = output_after

# Normalize features to calculate Cosine Similarity
features_before = features_before / features_before.norm(p=2, dim=-1, keepdim=True)
features_after = features_after / features_after.norm(p=2, dim=-1, keepdim=True)

# Calculate Cosine Similarity (1.0 = Identical, 0.0 = Completely Different)
similarity = (features_before @ features_after.T).item()
similarity_percentage = similarity * 100

# 4. Dispute Resolution Logic
print("\n📝 --- DISPUTE RESOLUTION SUMMARY ---")
print(f"Visual Similarity Score : {similarity_percentage:.1f}%")

if similarity_percentage < 40.0:
    verdict = "✅ WORK VERIFIED: Significant visual change detected between 'Before' and 'After' photos."
    recommendation = "Recommendation: Approve wage payment. The evidence suggests the task was completed."
elif similarity_percentage < 75.0:
    verdict = "⚠️ PARTIAL COMPLETION: Moderate visual change detected."
    recommendation = "Recommendation: Flag for manual review. Minor adjustments may have been made, but major work is unclear."
else:
    verdict = "❌ WORK NOT VERIFIED: Photos are nearly identical."
    recommendation = "Recommendation: Reject wage claim. Insufficient evidence of task completion."

print(f"AI Verdict            : {verdict}")
print(f"System Recommendation : {recommendation}")
print("------------------------------------------")
print("💡 Presentation Note: CLIP provides a robust, language-agnostic way to verify")
print("   physical task completion without needing custom-trained object detection models.\n")