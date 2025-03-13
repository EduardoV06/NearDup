from PIL import Image
import torchvision.transforms as transforms


def load_and_preprocess_image(image_path):
    """Load and preprocess an image for torchvision models."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def estimate_image_memory(image_path):
    """Estimate the image size on memory"""
    img = Image.open(image_path)
    width, height = img.size
    bytes_per_pixel = len(img.getbands())  # RGB = 3, RGBA = 4, etc.
    estimated_size = width * height * bytes_per_pixel  # Em bytes
    return estimated_size

def process_image_safely(image_path, max_size=50 * 1024 * 1024):  # 50MB limite
    size = estimate_image_memory(image_path)
    
    if size > max_size:
        print(f"Imagem {image_path} é muito grande ({size / (1024*1024):.2f} MB). Processando de forma lazy...")
        # Processamento alternativo (ex: carregamento parcial)
        return Image.open(image_path).resize((512, 512))  # Reduz a resolução
    
    return Image.open(image_path)  # Carrega normalmente