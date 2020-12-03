#!/usr/bin/env python3
# -*-*- encoding: utf-8 -*-*-
# Created: Sat, 26 May 2018 19:32:23 -0300

"""
Detector de faces usando a biblioteca OpenCV.

Uso: detector_facial.py [-c CLASSIFICADOR]

Argumentos opcionais:
  -c CLASSIFICADOR, --classificador CLASSIFICADOR
                        Caminho para classificador em cascata
"""

import os
import sys
import argparse
import logging
import cv2


__author__ = "Julio Batista Silva"
__copyright__ = "Copyright (c) 2018, Julio Batista Silva"
__license__ = "GPL v3"
__version__ = "1.0"
__email__ = "julio@juliobs.com"


SCRIPT_DIR = os.path.dirname(__file__)
SCRIPT_NAME = os.path.basename(__file__)
DEFAULT_CLASSIFIER = 'haarcascades/haarcascade_frontalface_default.xml'


def detecta_faces(imagem, classificador):
    ''' Retorna uma lista com as coordenadas das faces detectadas na imagem
    pelo classificador passado por parâmetro'''
    # Carrega o classificador em cascata passado por parâmetro
    classif = cv2.CascadeClassifier(classificador)

    # Cria cópia em tons de cinza da imagem
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Detecta as faces usando a imagem em tons de cinza
    faces = classif.detectMultiScale(
        cinza, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

    return faces


def main(ARGS):

    for img_path in ARGS.imagens:
        imagem = cv2.imread(img_path)

        if imagem is None:
            continue

        faces = detecta_faces(imagem, ARGS.classificador)

        if len(faces) > 0:
            # Desenha um retângulo em volta das faces detectadas
            for (x, y, w, h) in faces:
                cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Grava imagem com o resultado da detecção
            pasta = ARGS.saida if ARGS.saida else os.path.dirname(img_path)
            nome = 'DETECT-' + os.path.basename(img_path)
            cv2.imwrite(os.path.join(pasta, nome), imagem)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detecta faces na câmera.')
    parser.add_argument('-c', '--classificador', type=str,
                        default=DEFAULT_CLASSIFIER,
                        help='Caminho para classificador em cascata')
    parser.add_argument('-i', '--imagens', nargs='*',
                        help='Caminho para imagem')
    parser.add_argument('-s', '--saida',
                        help='Caminho onde detecções serão salvas')
    parser.add_argument('--versao', action='version',
                        version='%(prog)s v' + __version__)
    ARGS = parser.parse_args()

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
