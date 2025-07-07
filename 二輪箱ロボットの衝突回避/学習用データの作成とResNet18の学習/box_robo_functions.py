from math import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import ode

from PIL import Image


#4つの方角（北東南西）の処理が終わったときに衝突判定の学習用クラスラベルデータを出力
def box_collision_class_label_file(class_label, txt_file_name):
    with open(txt_file_name,"w") as o:
        for index, v in enumerate(class_label):
            if index == len(class_label) - 1:
                print(str(v), end="", file=o)
            else:
                print(str(v) + ",", end="", file=o)

#無効座標ファイル作成用☆☆☆#
def invalid_position_matrix_file(invalid_position_matrix, txt_file_name):
    with open(txt_file_name,"w") as o:
        # ループで全要素を取得
        for row in invalid_position_matrix:
            for index, element in enumerate(row):
                if index == len(row) - 1:
                    print(str(element), file=o)  
                else:                                  
                    print(str(element) + ",", end=" ", file=o)

def draw_body(body):
    """Draw an ODE body.
    """
    x,y,z = body.getPosition()
    R = body.getRotation()
    rot = [R[0], R[3], R[6], 0.,
        R[1], R[4], R[7], 0.,
        R[2], R[5], R[8], 0.,
        x, y, z, 1.0]
    glPushMatrix()
    glMultMatrixd(rot)

    if body.shape=="box":
        sx,sy,sz = body.boxsize
        glScalef(sx, sy, sz)
        glutSolidCube(1)

    if body.shape=="cylinder":
        #ODEのcylinder（重心座標）とOpenGLのcylinder（ボトム座標）の座標設定の違いを修整して描画する。

        #シリンダーの外周を作成
        r,h = body.cylindersize
        R=[1., 0., 0., 0., 1, 0, 0., 0, 1]#x軸周りに0度回転
        rot = [R[0], R[3], R[6], 0.,
            R[1], R[4], R[7], 0.,
            R[2], R[5], R[8], 0.,
            0, 0, -h*0.5, 1.0]  #z軸方向に -h*0.5 平行移動
        glMultMatrixd(rot)
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluCylinder(quadric, r, r, h, 10, 10)  # Base radius, top radius, height, slices, stacks    
        gluDeleteQuadric(quadric)
        
        #シリンダーの上下2枚のディスクを作成
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluDisk(quadric, 0, r, 10, 10)  # Inner radius, outer radius, slices, loops
        rot = [R[0], R[3], R[6], 0.,
            R[1], R[4], R[7], 0.,
            R[2], R[5], R[8], 0.,
            0, 0, h, 1.0]   #z軸方向に h 平行移動

        glMultMatrixd(rot)
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluDisk(quadric, 0, r, 10, 10)  # Inner radius, outer radius, slices, loops
        gluDeleteQuadric(quadric)
        
    glPopMatrix()

# create_box
def create_box(world, space, density, lx, ly, lz):
    """Create a box body and its corresponding geom."""
    # Create body
    body = ode.Body(world)
    M = ode.Mass()
    M.setBox(density, lx, ly, lz)
    body.setMass(M)

    # Set parameters for drawing the body
    body.shape = "box"
    body.boxsize = (lx, ly, lz)

    # Create a box geom for collision detection
    geom = ode.GeomBox(space, lengths=body.boxsize)
    geom.setBody(body)
    return body, geom

# create_cylinder
def create_cylinder(world, space, density, direction, r, h):
    """Create a cylinder body and its corresponding geom."""
    # Create body
    body = ode.Body(world)
    M = ode.Mass()
    M.setCylinder(density, direction, r, h)
    body.setMass(M)

    # Set parameters for drawing the body
    body.shape = "cylinder"
    body.cylindersize = (r, h)
    
    # Create a box geom for collision detection
    geom = ode.GeomCylinder(space, r, h)
    geom.setBody(body)
    return body, geom 

# drop_box_object
def drop_box( world, space, bodies, geoms, objcount_tmp, lx, ly, lz, px, py, pz, density):
    """Drop an object into the scene."""
    #global bodies, geoms, objcount

    body, geom = create_box(world, space, density, lx, ly, lz)
    theta = 0
    body.setPosition( (px, py, pz) )
    ct = cos (theta)
    st = sin (theta)
    body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])#y軸回転
    #body.setRotation([1., 0., 0., 0., ct, -st, 0., st, ct])#x軸回転 

    bodies.append(body)
    geoms.append(geom)
    #objcount += 1
    objcount_tmp[0] += 1

# drop_cylinder_object
def drop_cylinder( world, space, bodies, geoms, objcount_tmp, rotation_num, r, h, px, py, pz, density):
    """Drop an object into the scene."""
    #global bodies, geoms, objcount

    body, geom = create_cylinder(world, space, density, 3, r, h)  #odeとopenglのシリンダーの方向を一致させるために、3(z軸方向)にする。
    
    if rotation_num == 1:
        theta = 3.1415*(0.0/180.0)  #シリンダーの方向はz軸方向
    if rotation_num == 2:
        theta = 3.1415*(90.0/180.0) #シリンダーの方向をy軸方向にして、シリンダーを立てる。

    body.setPosition( (px, py, pz) )  
    ct = cos (theta)
    st = sin (theta)

    body.setRotation([1., 0., 0., 0., ct, st, 0., -st, ct])#x軸回転
    #body.setRotation([ct, 0., -st, 0., 1., 0., st, 0., ct])#y軸回転
    #body.setRotation([ct, st., 0., -st, ct, 0., 0., 0., 1.])#z軸回転

    bodies.append(body)
    geoms.append(geom)

    #objcount += 1
    objcount_tmp[0] += 1


#テクスチャ読み込み関数
def load_texture(texture_file_name):

    # Generate a texture ID
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id )

    img = Image.open(texture_file_name)
    w, h = img.size
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.tobytes())
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)    # Set texture parameters
    glBindTexture(GL_TEXTURE_2D, 0)  # Unbind texture

    return texture_id

#テクスチャポリゴン
def draw_tex_polygon(tex_floor, tex_wall):

    glMaterialfv(GL_FRONT, GL_AMBIENT, [0.5, 0.5, 0.5, 0.5])  #環境光の影響  
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])#地の色の設定

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)#テクスチャの色と地の色の混ざり方の設定
    glEnable(GL_BLEND)
    glEnable(GL_TEXTURE_2D)
    glNormal3f(0, 1, 0)     #glNormal3f()は非推奨の関数らしい

    #床
    glBindTexture(GL_TEXTURE_2D, tex_floor, )
    glBegin(GL_QUADS)

    glTexCoord2d(4.5, 0.0)
    glVertex3d(4.5, -0.01, -4.5)

    glTexCoord2d(0.0, 0.0)
    glVertex3d( -4.5, -0.01, -4.5)

    glTexCoord2d(0.0, 4.5)
    glVertex3d( -4.5, -0.01, 4.5)

    glTexCoord2d(4.5, 4.5)
    glVertex3d(4.5, -0.01, 4.5)

    glEnd()

    #壁　奥
    glBindTexture(GL_TEXTURE_2D, tex_wall)
    glBegin(GL_QUADS)

    glTexCoord2d(1.0, 0.0)
    glVertex3d(4.5,  0.0, -4.5)

    glTexCoord2d(0.0, 0.0)
    glVertex3d( -4.5, 0.0, -4.5)

    glTexCoord2d(0.0, 1.0)
    glVertex3d( -4.5, 2.0, -4.5)

    glTexCoord2d(1.0, 1.0)
    glVertex3d(4.5,  2.0, -4.5)

    glEnd()
    
    #壁　左
    glBegin(GL_QUADS)

    glTexCoord2d(1.0, 0.0)
    glVertex3d(-4.5, 0.0, 4.5)

    glTexCoord2d(0.0, 0.0)
    glVertex3d( -4.5, 0.0, -4.5)

    glTexCoord2d(0.0, 1.0)
    glVertex3d( -4.5, 2.0, -4.5)

    glTexCoord2d(1.0, 1.0)
    glVertex3d(-4.5,  2.0, 4.5)

    glEnd()
    
    #壁　手前
    glBegin(GL_QUADS)

    glTexCoord2d(1.0, 0.0)
    glVertex3d(4.5,  0.0, 4.5)

    glTexCoord2d(0.0, 0.0)
    glVertex3d( -4.5, 0.0, 4.5)

    glTexCoord2d(0.0, 1.0)
    glVertex3d( -4.5, 2.0, 4.5)

    glTexCoord2d(1.0, 1.0)
    glVertex3d(4.5,  2.0, 4.5)

    glEnd()
    
    #壁　右
    glBegin(GL_QUADS)

    glTexCoord2d(1.0, 0.0)
    glVertex3d(4.5, 0.0, 4.5)

    glTexCoord2d(0.0, 0.0)
    glVertex3d(4.5, 0.0, -4.5)

    glTexCoord2d(0.0, 1.0)
    glVertex3d(4.5, 2.0, -4.5)

    glTexCoord2d(1.0, 1.0)
    glVertex3d(4.5, 2.0, 4.5)

    glEnd()
    
    glDisable(GL_TEXTURE_2D)

    glDisable(GL_BLEND)

    glFlush()#??#


def room3_1( world, space, bodies, geoms, objcount_tmp):
            
    #ベッドの天板
    drop_box( world, space, bodies, geoms, objcount_tmp, 2.2, 0.2, 1.2,    
            1.0, 0.3, -3.5, 1000)  #(lx, ly, lz, px, py, pz, density)   
    #ベッドの足#####
    drop_cylinder( world, space, bodies, geoms, objcount_tmp, 2, 0.1, 0.2,     
            0., 0.1, -4., 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
    drop_cylinder( world, space, bodies, geoms, objcount_tmp, 2, 0.1, 0.2,     
            2., 0.1, -4., 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
    drop_cylinder( world, space, bodies, geoms, objcount_tmp, 2, 0.1, 0.2,     
            2., 0.1, -3., 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
    drop_cylinder( world, space, bodies, geoms, objcount_tmp, 2, 0.1, 0.2,     
            0., 0.1, -3., 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
    
    #扇風機の頭1
    drop_cylinder( world, space, bodies, geoms, objcount_tmp, 1, 0.3, 0.1,     
            0, 0.7, -0.25 + 2.0, 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
    #扇風機の頭2
    drop_cylinder( world, space, bodies, geoms, objcount_tmp, 1, 0.1, 0.4,     
            0, 0.7, 0. + 2.0, 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
    #扇風機の足1
    drop_cylinder( world, space, bodies, geoms, objcount_tmp, 2, 0.4, 0.1,     
            0, 0.05, 0. + 2.0, 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。
    #扇風機の足2
    drop_cylinder( world, space, bodies, geoms, objcount_tmp, 2, 0.1, 0.6,     
            0, 0.3, 0. + 2.0, 1000)  #(rotation_num, r, h, px, py, pz, density) 位置座標はボトムではなく重心の座標に変換している。

    #机の天板
    drop_box( world, space, bodies, geoms, objcount_tmp, 1.3, 0.2, 1.7,    
            3.8, 0.7, 0.825, 1000)  #(lx, ly, lz, px, py, pz, density)
    #机の脚
    drop_box( world, space, bodies, geoms, objcount_tmp, 0.15, 0.6, 0.15, 3.3, 0.3, 0., 1000)  #(lx, ly, lz, px, py, pz, density)  
    drop_box( world, space, bodies, geoms, objcount_tmp, 0.15, 0.6, 0.15, 4.3, 0.3, 0., 1000)  #(lx, ly, lz, px, py, pz, density)  
    #机の引き出し
    drop_box( world, space, bodies, geoms, objcount_tmp, 1.0, 0.6, 0.5,    
            3.8, 0.3, 1.4, 1000)  #(lx, ly, lz, px, py, pz, density)     

    #棚の天板1
    drop_box( world, space, bodies, geoms, objcount_tmp, 1.0, 0.1, 2.0, 0.5 - 4.2, 0.6, 1.0, 1000)  #(lx, ly, lz, px, py, pz, density)  
    #棚の天板2
    drop_box( world, space, bodies, geoms, objcount_tmp, 1.0, 0.1, 2.0, 0.5 - 4.2, 1.0, 1.0, 1000)  #(lx, ly, lz, px, py, pz, density)  
    #棚の脚
    drop_box( world, space, bodies, geoms, objcount_tmp, 0.15, 1., 0.15, 0. - 4.2, 0.5, 0., 1000)  #(lx, ly, lz, px, py, pz, density)  
    drop_box( world, space, bodies, geoms, objcount_tmp, 0.15, 1., 0.15, 1. - 4.2, 0.5, 0., 1000)  #(lx, ly, lz, px, py, pz, density)  
    drop_box( world, space, bodies, geoms, objcount_tmp, 0.15, 1., 0.15, 1. - 4.2, 0.5, 2., 1000)  #(lx, ly, lz, px, py, pz, density)  
    drop_box( world, space, bodies, geoms, objcount_tmp, 0.15, 1., 0.15, 0. - 4.2, 0.5, 2., 1000)  #(lx, ly, lz, px, py, pz, density)      
    
    #外周
    drop_box( world, space, bodies, geoms, objcount_tmp, 0.3, 1.0, 8.99,
                -4.651, 0.5001, 0.001, 1000)  #(lx, ly, lz, px, py, pz, density)
    drop_box( world, space, bodies, geoms, objcount_tmp, 0.3, 1.0, 8.99,
                4.651, 0.5001, 0.001, 1000)  #(lx, ly, lz, px, py, pz, density)
    drop_box( world, space, bodies, geoms, objcount_tmp, 8.99, 1.0, 0.3,
                0.001, 0.5001, 4.651, 1000)  #(lx, ly, lz, px, py, pz, density)
    drop_box( world, space, bodies, geoms, objcount_tmp, 8.99, 1.0, 0.3,
                0.001, 0.5001, -4.651, 1000)  #(lx, ly, lz, px, py, pz, density)
    