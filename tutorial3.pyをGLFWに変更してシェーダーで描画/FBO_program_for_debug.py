import sys, os, random, time
from math import *
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
import glfw
import glm
import numpy as np

import ode

from pyrr import Matrix44, Vector3

# Vertex Shader
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
void main() {
    gl_Position = vec4(aPos, 1.0);
}
"""

# Fragment Shader
fragment_shader_source = """
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 0.0, 0.0, 1.0); // 赤色
}
"""

#シェーダコンパイル
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    # Check compilation
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile failed: {error}")
    return shader

#シェーダプログラム作成
def create_shader_program():
    vertex_shader = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    # Check linking
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program linking failed: {error}")
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

######################################################################

# Initialize GLFW
glfw.init()

# Create Window
window = glfw.create_window(800, 600, "PyOpenGL GLFW drop box", None, None)
if not window:
    glfw.terminate()
    #return
glfw.make_context_current(window)

def main():

    #シャドウマッピング用にフレームバッファと深度テクスチャを設定。
    # Create a framebuffer
    FBO = glGenFramebuffers(1)  #
    glEnable(GL_DEPTH_TEST)     # Enable depth test for 3D rendering
    glDepthMask(GL_TRUE)        #
    glDepthFunc( GL_LEQUAL )    #
    #深度テクスチャ分の領域の生成
    depth_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_texture )# テクスチャ0にバインド
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 800, 600, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None) #CPUを使わないようにするため、Noneを指定して転送しない
    #glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 800, 600, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    # テクスチャパラメータの設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glBindTexture(GL_TEXTURE_2D, 0)  # Unbind texture
    #FBOにバインド
    glBindFramebuffer(GL_FRAMEBUFFER, FBO)  
    #深度テクスチャ分の領域をFBOフレームバッファにアタッチ
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0)
    #glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depth_texture, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    #デフォルトフレームバッファ（画面のバッファ）にバインド
    glBindFramebuffer(GL_FRAMEBUFFER, 0) 
    
    #
    shader_program = create_shader_program()

    # Generate buffers and arrays
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    fl_count = 0
    #物理演算とシェーダのループ部分
    while not glfw.window_should_close(window):
        fl_count += 1

        #頂点のデータ
        vertices = np.array([], dtype=np.float32)
        #三角形の頂点の番号
        indices = np.array([], dtype=np.uint32)

        #床と壁の頂点データを作成
        #床と壁のverticesデータを作成
                            # positions
        floor_arr = np.array([  -1.0,  1.0,  -1.0, #床 0
                                -1.0,  0.0,   1.0, #床 1
                                1.0,  0.0,  -1.0,  #床 2
                                1.0,  0.0,   1.0  #床 3
                            ], dtype=np.float32)   
        vertices = np.append(vertices, floor_arr)   #頂点のデータ
        #床と壁のIndicesデータを作成
        if len(indices) == 0:
            i = 0
        else:
            i = max(indices) + 1
        arr3 = np.array([0+i,1+i,2+i,  3+i,1+i,2+i,  #床
                        ], dtype=np.uint32)
        indices = np.append(indices, arr3)  #三角形の頂点の番号

        #
        glBindVertexArray(VAO)

        # Vertex buffer
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Element buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        #
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)


        # Start render 
        glClearColor(0.2, 0.3, 0.3, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader_program)

        glBindFramebuffer(GL_FRAMEBUFFER, FBO)  #
        glViewport(0, 0, 800, 600)
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        #
        data = glReadPixels(0, 0, 800, 600, GL_DEPTH_COMPONENT, GL_FLOAT)
        flat_data = [el for row in data for el in row]
        print(max(flat_data))
        print(min(flat_data))

        if fl_count > 1:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)  #デフォルトフレームバッファにバインド

        glfw.swap_buffers(window)
        glfw.poll_events()

    # Cleanup
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteBuffers(1, [EBO])
    glDeleteBuffers(1, [FBO])
    glfw.terminate()

if __name__ == "__main__":
    main()