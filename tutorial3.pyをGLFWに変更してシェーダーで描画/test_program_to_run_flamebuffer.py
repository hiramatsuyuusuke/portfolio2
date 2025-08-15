import glfw
from OpenGL.GL import *
import numpy as np

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW initialization failed")

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(800, 600, "PyOpenGL with GLFW", None, None)
if not window:
    glfw.terminate()
    raise Exception("Failed to create GLFW window")

# Make the window's context current
glfw.make_context_current(window)

# Create a framebuffer
framebuffer = glGenFramebuffers(1)
glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

# Create a texture to attach to the framebuffer
texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 800, 600, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

# Attach the texture to the framebuffer
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

# Check if the framebuffer is complete
if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
    raise Exception("Framebuffer is not complete")

# Unbind the framebuffer
glBindFramebuffer(GL_FRAMEBUFFER, 0)

# Main loop
while not glfw.window_should_close(window):
    # Render to the framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
    glClearColor(1.0, 0.0, 0.0, 1.0)  # Red background
    glClear(GL_COLOR_BUFFER_BIT)

    # Render to the default framebuffer (screen)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glClearColor(0.0, 1.0, 0.0, 1.0)  # Black background
    glClear(GL_COLOR_BUFFER_BIT)

    # Draw the texture (optional, for debugging)
    glBindTexture(GL_TEXTURE_2D, texture)
    # Add rendering code here if needed

    glfw.swap_buffers(window)
    glfw.poll_events()

# Cleanup
glDeleteTextures([texture])
glDeleteFramebuffers(1, [framebuffer])
glfw.terminate()
