#!/usr/bin/env python2
# -*-*- encoding: utf-8 -*-*-
# Created: Thu, 14 Jun 2018 22:33:51 -0300

"""
Detector de faces usando a biblioteca OpenCV no Raspberry Pi.

Uso: detector_facial-raspberry.py [-c CLASSIFICADOR]

Argumentos opcionais:
  -c CLASSIFICADOR, --classificador CLASSIFICADOR
                        Caminho para classificador em cascata
  -d DIR_FOTOS, --dir_fotos DIR_FOTOS
                        Caminho do diretório onde as fotos serão salvas
"""

import os
import sys
import argparse
import logging
import time
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray


__author__ = "Julio Batista Silva"
__copyright__ = "Copyright (c) 2018, Julio Batista Silva"
__license__ = "GPL v3"
__version__ = "1.0"
__email__ = "julio@juliobs.com"


SCRIPT_DIR = os.path.dirname(__file__)
SCRIPT_NAME = os.path.basename(__file__)


DEFAULT_CLASSIFIER = 'haarcascades/haarcascade_frontalface_default.xml'
DEFAULT_OUTPUT_PATH = 'faces/'


def detecta_faces(imagem, classificador):
    '''Retorna uma lista com a posição das faces detectadas usando o
    classificador passado por parâmetro'''

    # Carrega o classificador em cascata passado por parâmetro
    cc = cv2.CascadeClassifier(classificador)
    faces = cc.detectMultiScale(
        imagem, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    return faces


def main():
    '''Função principal'''

    # Detecta faces até interrupção
    with PiCamera(resolution=(640, 480), framerate=24) as camera:

        # Inverte câmera
        camera.rotation = 180

        # Tempo para esquentar câmera
        time.sleep(2)

        img_buffer = PiRGBArray(camera, size=camera.resolution)

        # Captura imagens
        for frame in camera.capture_continuous(img_buffer, format="bgr"):
            # Horário da captura
            horario = time.strftime("%Y%m%d%H%M%S")

            # Foto como um array
            imagem = frame.array

            # Cria cópia em tons de cinza da imagem
            cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

            # Detecta as faces usando a imagem em tons de cinza e o
            # classificador
            faces = detecta_faces(cinza, ARGS.classificador)

            # Detectou alguma face?
            if len(faces) > 0:
                logging.debug("================Face detectada================")

                # Contador de faces na imagem
                face_num = 0

                for (x, y, w, h) in faces:
                    # Numera face
                    face_num += 1

                    # Corta face detectada
                    face = imagem[y:y+h, x:x+w]

                    # Salva imagem cortada
                    cv2.imwrite(
                        os.path.join(
                            ARGS.dir_fotos,
                            "{}_{}.jpg".format(horario, face_num)),
                        face)

            # Esvazia captura para reutilizar no próximo frame
            img_buffer.truncate(0)

            # Espera 1s antes de capturar outra foto
            time.sleep(1)

    return 0


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='Detecta faces na câmera.')
    PARSER.add_argument('-c', '--classificador', dest="classificador",
                        default=DEFAULT_CLASSIFIER,
                        help='Caminho para classificador em cascata')
    PARSER.add_argument('-d', '--dir_fotos', dest="dir_fotos",
                        default=DEFAULT_OUTPUT_PATH,
                        help='Caminho do diretório onde as fotos serão salvas')
    PARSER.add_argument('--versao', action='version',
                        version='%(prog)s v' + __version__)
    ARGS = PARSER.parse_args()

    logging.basicConfig(
        filename=os.path.join(SCRIPT_DIR, SCRIPT_NAME + '.log'),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')

    logging.debug("===============Iniciou o script===============")

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.debug("=================Interrompido=================")
        sys.exit(0)
    except SystemExit:
        logging.debug("==============Finalizou o script==============")
    except BaseException:
        logging.exception('Ocorreu um erro')
