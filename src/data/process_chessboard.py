import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import random


RESIZED_IMAGE_HEIGHT = 256
RESIZED_IMAGE_WIDTH = 256
NUM_SQUARES_IN_ROW = 8
NUM_SQUARES_IN_COL = 8
NUM_VALUES_PER_SQUARE = 32 * 32 * 3
CHESSBOARD_COLS = list('abcdefgh')
# CLASS_SELECTION_PROB = {'p': 0.3, 'P': 0.3, 'E': 0.025}
CLASS_SELECTION_PROB = {}

def chessboard_squares_gen(output_file_str, img):
    # Break a chessboard image into squares and generate a class for each
    square_width = RESIZED_IMAGE_WIDTH // NUM_SQUARES_IN_ROW
    square_height = RESIZED_IMAGE_HEIGHT // NUM_SQUARES_IN_COL
    for j, y in enumerate(range(0, RESIZED_IMAGE_HEIGHT, square_height)):
        for i, x in enumerate(range(0, RESIZED_IMAGE_WIDTH, square_width)):
            square = img[y:y + square_height, x:x + square_width]
            square_id = "{0}{1}".format(CHESSBOARD_COLS[i], NUM_SQUARES_IN_ROW - j)
            # print(CHESSBOARD_COLS[i], NUM_SQUARES_IN_ROW - j, y, y + square_height, x, x + square_width,
            #       square[0][0], square[10][5])
            cv2.imwrite(output_file_str.format(square_id), square)
            yield square_id, square


def prepare_image(img_file_name):
    img = cv2.imread(img_file_name)
    resized_img = cv2.resize(img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
    return resized_img


def get_target_from_fen(fen_str):
    # Get a target class for each square from the fen
    fen_target_map = {}
    fen_info = fen_str.split(' ')
    rank = NUM_SQUARES_IN_COL
    for rank_val in fen_info[0].split('/'):
        file_idx = 0
        idx = 0
        curr_val = None
        while idx < len(rank_val):
            square_id = "{0}{1}".format(CHESSBOARD_COLS[file_idx], rank)
            if rank_val[idx].isnumeric():
                if not curr_val:
                    curr_val = int(rank_val[idx])
                fen_target_map[square_id] = 'E'
                curr_val -= 1
                if curr_val <= 0:
                    idx += 1
            else:
                fen_target_map[square_id] = rank_val[idx]
                idx += 1
            file_idx = (file_idx + 1) % NUM_SQUARES_IN_ROW
        rank -= 1
    return fen_target_map


input_dir_name = "../../data/raw/raw-images"
output_image_file_str = "../../data/interim/chessboard_squares/b00-kings-pawn-game-lichess-{0}.png"
name_fen_map = "../../data/raw/name-fen-map.csv"
features_file_name = "../../data/processed/chessboard_squares.txt"
targets_file_name = "../../data/processed/chessboard_square_classes.txt"
features_file = open(features_file_name, 'w')
targets_file = open(targets_file_name, 'w')
with open(name_fen_map) as csvfile:
    name_fen_map_reader = csv.reader(csvfile, delimiter=',')
    next(name_fen_map_reader)
    for row in name_fen_map_reader:
        image_file_name = os.path.join(input_dir_name, row[0].strip())
        fen_str = row[1].strip()
        print(image_file_name, ':', fen_str)
        target_map = get_target_from_fen(fen_str)
        # print(target_map)
        img = prepare_image(image_file_name)
        plt.imshow(img)
        plt.show()
        for square_id, square in chessboard_squares_gen(output_image_file_str, img):
            # print(square.shape)
            square_data = square.reshape(1, NUM_VALUES_PER_SQUARE)
            # print(square_data.shape)
            r = random.random()
            if target_map[square_id] not in CLASS_SELECTION_PROB or r < CLASS_SELECTION_PROB[target_map[square_id]]:
                np.savetxt(features_file, square_data, delimiter=' ', fmt="%d")
                targets_file.write("{0}\n".format(target_map[square_id]))
            print(square_id, len(square_data), square_data, target_map[square_id], r)
