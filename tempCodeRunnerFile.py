model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
# model.eval()

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def predict(image):
#     with torch.no_grad():
#         image_tensor = transform(image).unsqueeze(0)
#         output = model(image_tensor)
#         return output