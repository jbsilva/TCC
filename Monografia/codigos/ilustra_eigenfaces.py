#!/usr/bin/env python3
# -*-*- encoding: utf-8 -*-*-
# Created: Sat, 23 Jun 2018 17:57:15 -0300

"""
Para ilustrar o eigenfaces
"""


import os
import sys
import argparse
import logging
import numpy as np
import cv2


__author__ = "Julio Batista Silva"
__copyright__ = "Copyright (c) 2018, Julio Batista Silva"
__license__ = "GPL v3"
__version__ = "1.0"
__email__ = "julio@juliobs.com"


SCRIPT_DIR = os.path.dirname(__file__)
SCRIPT_NAME = os.path.basename(__file__)


def normaliza(arr):
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3)


def main():
    '''Calcula Eigenfaces usando OpenCV'''

    # Lista de imagens
    #from glob import glob
    #arquivos = glob(f"{faces_dir}/*.jpg")
    arquivos = ARGS.imagens
    qtd_imgs = len(arquivos)

    # Cria lista de imagens representadas como array de floats entre 0 e 1
    imagens = []
    for arquivo in arquivos:
        img = cv2.imread(arquivo)
        img = np.float32(img) / 255.0
        imagens.append(img)

    # Dimensões das imagens
    dims = imagens[0].shape

    # Passa todas as imagens para arrays de uma dimensão
    fotos = np.zeros((qtd_imgs, dims[0] * dims[1] * dims[2]), dtype=np.float32)
    for i in range(qtd_imgs):
        fotos[i, :] = imagens[i].flatten()

    # Calcula a média e os autovetores
    #qtd_autov = min(200, qtd_imgs)
    media, autovetores = cv2.PCACompute(
        fotos, mean=None)
        #fotos, mean=None, maxComponents=qtd_autov)

    # Face média é o vetor médio em forma matricial
    face_media = media.reshape(dims)
    cv2.imwrite("face_media.png", face_media * 255)

    # Eigenfaces são os autovetores em forma matricial
    eigenfaces = []
    for i, autovetor in enumerate(autovetores):
        eigenface = autovetor.reshape(dims)
        eigenfaces.append(eigenface)
        cv2.imwrite(f"eigenface_{i:03}.png", normaliza(eigenface))

    # Reconstrói uma faces a partir das eigenfaces
    foto = np.float32(cv2.imread(ARGS.reconstruir)) / 255.0
    face_norm = foto.flatten() - media
    face_reco = face_media
    coeficientes = []
    for i, autovetor in enumerate(autovetores):
        coef = np.dot(face_norm, autovetor)
        coeficientes.append(coef.item())
        face_reco = np.add(face_reco, eigenfaces[i] * coef)
        cv2.imwrite(f"reconstruida_{i:03}.png", face_reco * 255)

    print ("Coeficientes:")
    for i, coef in enumerate(coeficientes):
        print(f"{i:03}: {coef}")

    return 0


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='Ilustra PCA e Eigenfaces')
    PARSER.add_argument('-i', '--imagens', nargs='*',
                        help='Caminhos das imagens de entrada')
    PARSER.add_argument('-r', '--reconstruir',
                        help='Caminhos da imagem a ser reconstruida')
    PARSER.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    ARGS = PARSER.parse_args()

    logging.basicConfig(
        filename=os.path.join(SCRIPT_DIR, SCRIPT_NAME + '.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    logging.debug("===================Iniciou o script===================")

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.debug("=====================Interrompido=====================")
        sys.exit(0)
    except SystemExit:
        logging.debug("==================Finalizou o script==================")
    except BaseException:
        logging.exception('Ocorreu um erro')
