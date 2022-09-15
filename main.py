import numpy as np
import pygame
import cv2
import matplotlib.image
import torch
import gc

LEFT = 1
RIGHT = 3

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

pygame.init()

clock = pygame.time.Clock()

# Set up the drawing window
screen = pygame.display.set_mode([700, 500])

font = pygame.font.SysFont('Comic Sans MS', 36)

brush = None
brush_size = 20
brush_color = (0, 0, 0)

canvas_ar = np.array([])

preds = []
pred = 0

canvas = pygame.Surface((500, 500))
canvas.fill((0, 0, 0))
canvas_rect = canvas.get_rect(topleft=(0, 0))

canvas_tensor = None

right_surface = pygame.Surface((200, 500))
right_surface.fill((150, 0, 0))
right_surface_rect = right_surface.get_rect(topleft=(500, 0))

text_surface = font.render("Pred: 0", True, (255, 255, 255))
text_surface_rect = text_surface.get_rect(topleft=(535, 200))

screen.blit(canvas, canvas_rect)
screen.blit(right_surface, right_surface_rect)
screen.blit(text_surface, text_surface_rect)

device = 'cpu'

model = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5)),
    torch.nn.MaxPool2d(2),
    torch.nn.ReLU(),
    
    torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(5, 5)),
    torch.nn.Dropout2d(),
    torch.nn.MaxPool2d(2),
    torch.nn.ReLU(),
    
    torch.nn.Flatten(),
    
    torch.nn.Linear(320, 50),
    torch.nn.ReLU(),
    
    torch.nn.Linear(50, 10),
    torch.nn.LogSoftmax(),
).to(device)

model.load_state_dict(torch.load("mnist_classifier_model.pth"))
model.eval()

start_time = pygame.time.get_ticks()

# Run until the user asks to quit
running = True
while running:
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == LEFT or event.button == RIGHT:
                brush = event.pos

                if event.button == LEFT:
                    brush_color = (255, 255, 255)
                else:
                    brush_color = (0, 0, 0)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == LEFT or event.button == RIGHT:
                brush = None

        elif event.type == pygame.MOUSEMOTION:
            if brush:
                brush = event.pos

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                canvas.fill((0, 0, 0))

            elif event.key == pygame.K_p:
                # matplotlib.image.imsave("img.png", canvas_ar, cmap="gray")
                # print(model)
                # print(canvas_tensor.unsqueeze(0).unsqueeze(0).shape)
                pass

    # draw brush in bufor
    if brush:
        pygame.draw.circle(canvas, brush_color, (brush[0], brush[1]), brush_size)

    
    current_time = pygame.time.get_ticks()
    if current_time - start_time > 100:
        canvas_ar = cv2.resize(np.round(rgb2gray(np.array(pygame.surfarray.array3d(canvas)))).T, dsize=(28, 28)) / 255
        canvas_ar = (canvas_ar - 0.1307) / 0.3081

        canvas_tensor = torch.from_numpy(canvas_ar.copy()).to(device).float().unsqueeze(0).unsqueeze(0)

        preds = model(canvas_tensor).detach()
        pred = np.argmax(preds.cpu()).item()

        start_time = current_time

    text_surface = font.render(f"Pred: {pred}", True, (255, 255, 255))
    text_surface_rect = text_surface.get_rect(topleft=(535, 200))

    screen.blit(canvas, canvas_rect)
    screen.blit(right_surface, right_surface_rect)
    screen.blit(text_surface, text_surface_rect)

    # Flip the display
    pygame.display.flip()

    clock.tick(60)

# Done! Time to quit.
pygame.quit()