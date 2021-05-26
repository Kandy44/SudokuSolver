import cv2
import pytesseract
import numpy as np

# Tested with sudokus generated from
# http://sudoku.becher-sundstroem.de/

# They allow the sudoku puzzle to be saved as pdf
# This code has been tested with image which has been cropped from pdf
# That contains id, level of difficulty, set number and unsolved sudoku puzzle

# This function solves the sudoku puzzle
def solver(puzzle, cellCount):
    # If all cells have been filled with numbers
    if cellCount > 80:
        # Return the final puzzle
        return puzzle

    # Getting current row and column number
    r, c = cellCount // 9, cellCount % 9

    # If current cell is not empty
    if puzzle[r][c] != '.':
        # Calling the function with incrementing the cellCount by 1
        return solver(puzzle, cellCount + 1)

    # Iterating over digits
    for num in "123456789":
        # If it is possible to add the current digit to puzzle
        if possibleStep(puzzle, num, r, c):
            # Add it to puzzle at current row and column
            puzzle[r][c] = num
            # If after adding the current digit, the sudoku is completed
            if solver(puzzle, cellCount + 1):
                # Return the puzzle
                return puzzle
            # Else, the current digit is not be added at current row and column
            # So we set it to .
            puzzle[r][c] = '.'

    # If the cell
    return False

# This function checks if current digit can be added at given row and column
def possibleStep(puzzle, num, r, c):
    # If current digit already exists in row, return False
    for row in range(9):
        if puzzle[r][row] == num:
            return False
    # If current digit already exists in column, return False
    for col in range(9):
        if puzzle[col][c] == num:
            return False

    cur_row, cur_col = (r//3) * 3, (c//3) * 3
    for sub_row in range(cur_row, cur_row+3):
        for sub_col in range(cur_col, cur_col + 3):
            if puzzle[sub_row][sub_col] == num:
                return False

    return True


def main():
    # Configuration for pytesseract
    xconfig = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'

    # Reading the image of unsolved sudoku puzzle
    test_img_path = 'sudoku2.png'
    image = cv2.imread(test_img_path)

    # Converting it to grayscale and applying canny filter
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)

    # Finding the contours in the canny filter applied image
    contours, hierarchy = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Sorting the contours in descending based on their area
    sortedContours = sorted(contours, key=lambda x: -cv2.contourArea(x))

    # Getting the bounds of the largest contour
    X, Y, W, H = cv2.boundingRect(sortedContours[0])
    
    # Storing the sudoku puzzle into an new image
    sudoku_img_extracted = image[Y:Y+H, X:X+W]
    # Storing an copy of above image
    unsolved_image = sudoku_img_extracted.copy()

    # These are the sizes of the sub blocks to be retrieved from sudoku puzzle
    windowsize_r = 50
    windowsize_c = 50

    # This grid stores the extracted digits from the sudoku puzzle
    grid = []

    # Iterating over sub blocks of image
    for r in range(0, sudoku_img_extracted.shape[0] - windowsize_r, windowsize_r):
        for c in range(0, sudoku_img_extracted.shape[1] - windowsize_c, windowsize_c):
            # Extracting the sub block
            window = sudoku_img_extracted[r:r+windowsize_r, c:c+windowsize_c]
            # Removing first 10 and last 10 rows from sub block
            window = window[10:]
            window = window[:-10]
            # Removing first 10 columns  from sub block
            window = np.delete(window, list(range(0, 11)), axis=1)

            # Converting sub block to grayscale
            gry = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            # Applying pytesseract on gray image with config
            txt = pytesseract.image_to_string(gry, config=xconfig)
            # If no number has been found
            if txt == '\x0c':
                # The current element is .
                txt = '.'
            # Else
            else:
                # Removing unnecessary values from text
                txt = txt.replace('\x0c', '')
                txt = txt.replace('\n', '')
            # Adding number to grid
            grid.append(txt)

    # Converting sudoku puzzle from 1D to 2D list
    sudoku_puzzle = np.reshape(grid, (9, 9))
    # Converting sudoku puzzle from numpy array to list
    sudoku_puzzle = [list(i) for i in sudoku_puzzle]
    
    # Creating an copy of unsolved puzzle
    unsolved_puzzle = [row[:] for row in sudoku_puzzle]
    # Solving the sudoku puzzle
    solved_sudoku_puzzle = solver(sudoku_puzzle, 0)

    # If sudoku puzzle cannot be solved, print a message
    if solved_sudoku_puzzle == False:
        print("The puzzle cannot be solved")
        return
    
    # constants for putText function
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 0)
    thickness = 2

    # These are the coordinates that represent where the digits are to be written
    c1, c2 = [15, 35]
    # Iterating over unsolved puzzle
    for i in range(9):
        c1 = 15
        for j in range(9):
            # If the current value in unsolved puzzle is .
            if unsolved_puzzle[i][j] == '.':
                # Then get it's equivalent digit from solved puzzle and write the digit onto the image
                sudoku_img_extracted = cv2.putText(sudoku_img_extracted, solved_sudoku_puzzle[i][j], (c1, c2), font,
                                                   fontScale, color, thickness, cv2.LINE_AA)
            # Increment the x by 50
            c1 += 50
        # Increment the y by 50
        c2 += 50
    
    # Showing the unsolved and solved sudoku puzzles
    cv2.imshow("Unsolved Sudoku", unsolved_image)
    cv2.imshow("Solved Sudoku", sudoku_img_extracted)

    # Updating the part of unsolved sudoku puzzle with solved puzzle in the original image
    image[Y:Y+H,X:X+W] = sudoku_img_extracted[:,:]
    
    # Saving the image
    cv2.imwrite('solved.jpg',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()