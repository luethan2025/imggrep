import imagehash
import PIL

def phash(img: PIL.Image.Image) -> imagehash.ImageHash:
    hash_value = imagehash.phash(img)
    return hash_value
