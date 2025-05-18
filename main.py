import cv2
import argparse
import easyocr
import numpy as np
import json
import matplotlib.pyplot as plt
import re
import pandas as pd
import difflib  # Used for string similarity comparison


class TextRecognition:

    def __init__(self):
        self.database = {}  # Define a dictionary to hold database entries

    def calculate_areas(self, word_coordinates_list):
        word_areas = []  # Initialize an empty list to store word areas
        for word_coordinates, _, _ in word_coordinates_list:
            # Extract the four corner points of the bounding box
            p1, p2, p3, p4 = word_coordinates

            # Calculate the width of the bounding box
            width = max(p1[0], p2[0], p3[0], p4[0]) - min(p1[0], p2[0], p3[0], p4[0])

            # Calculate the height of the bounding box
            height = max(p1[1], p2[1], p3[1], p4[1]) - min(p1[1], p2[1], p3[1], p4[1])

            # Calculate the area of the bounding box
            area = width * height

            word_areas.append(area)

        return word_areas

    def select_and_combine_options(self, input_array):
        selected_indices = []
        selected_options = []

        for i, option in enumerate(input_array):
            if ":" in option:
                selected_indices.append(i)
                selected_options.extend([option, input_array[i + 1]])

        print("Select options (separated by commas):")
        for i in range(0, len(selected_options), 2):
            print(f"{i // 2 + 1}. {selected_options[i]}")

        choices = input("Enter the numbers of your choices (comma-separated): ")
        choice_nums = [int(num.strip()) for num in choices.split(",")]

        new_array = []
        for num in choice_nums:
            new_array.extend([selected_options[(num - 1) * 2], selected_options[(num - 1) * 2 + 1]])

        return new_array

    def custom_sort(self, item):
        if isinstance(item, str):
            return item
        else:
            return str(item)

    def calculate_similarity_threshold(self, word):
        n = len(word)
        if n != 0:
            return int((n - 1) / n)

    def aml(self, input_array):
        cleaned_array = []
        for item in input_array:
            # Remove special characters and spaces, except for ":" and "-"
            cleaned_item = re.sub(r'[^\w/: -]', '', item)

            # Format dates enclosed in double quotes
            if re.match(r'"\d{2}/\d{2}/\d{4}"', cleaned_item):
                cleaned_item = cleaned_item[1:-1]  # Remove double quotes
            if cleaned_item.strip() != "":  # Check if the cleaned item is not empty after stripping
                cleaned_array.append(cleaned_item)
        return cleaned_array

    def binary_search(self, word, word_list):
        left, right = 0, len(word_list) - 1
        while left <= right:
            mid = (left + right) // 2
            mid_word = str(word_list[mid])

            if mid_word == word:
                return True
            elif mid_word < word:
                left = mid + 1
            else:
                right = mid - 1

        return False

    def clean_and_format_array(self, input_array):
        cleaned_array = []
        for item in input_array:
            cleaned_item = re.sub(r'[^\w/: -]', '', item)

            if re.match(r'"\d{2}/\d{2}/\d{4}"', cleaned_item):
                cleaned_item = cleaned_item[1:-1]
            cleaned_array.append(cleaned_item)
        return cleaned_array

    def find_selected_country(self, all, all_areas, database):
     exp_country = []  # Initialize an empty list to store detected country names
     ar_country = []   # Initialize an empty list to store corresponding areas

     for i in range(len(all)):
        country_found = self.binary_search(all[i], database[0])
        if country_found:
            exp_country.append(all[i])     # Append the detected country name
            ar_country.append(all_areas[i]) # Append the corresponding area

     if not ar_country:
        return "none"  # No countries detected, return "none"

     ir = "ireland"
     if ir in exp_country:
        ir_index = exp_country.index(ir)  # Get the index of "ireland" in the list
        exp_country.pop(ir_index)  # Remove "ireland" from the list of detected countries
        ar_country.pop(ir_index)  # Remove the corresponding area as well
     else:
        ir = None  # If "ireland" is not detected, set it to None

     if len(exp_country) > 0:
        # Find the index of the country with the largest area among the remaining countries
        max_area_index = ar_country.index(max(ar_country))
        selected_country = exp_country[max_area_index]
     else:
        # If no countries other than "ireland" are detected, return "ireland"
        selected_country = "ireland"

     return selected_country

    def find_selected_fruit(self, all, all_areas, database):
        selected_country = "none"
        max_area = 0  # Initialize max_area to 0
        for i in range(len(all)):
            country_found = self.binary_search(all[i], database[2])
            if country_found:
                # Check if the current area is greater than the max_area
                if all_areas[i] > max_area:
                    max_area = all_areas[i]
                    selected_country = all[i]
        return selected_country

    def similarity(self, a, b):
        if isinstance(a, str) and isinstance(b, str):
            return difflib.SequenceMatcher(None, a, b).ratio()
        else:
            return 0.0

    def extract_weight(self, words):
     weight = "n/a"
     preference_order = ["g", "ge", "kg", "9"]

     for preferred_ending in preference_order:
        for word in words:
            if word.endswith(preferred_ending):
                valid = True
                digits = word[:-len(preferred_ending)]  # Get the part of the word without the preferred ending
                if not digits or not digits.replace("-", "").isdigit():
                    valid = False
                if valid:
                    weight = digits + preferred_ending
                    return weight

     return weight


    def extract_large_numbers(self, input_list):
        for element in input_list:
            digits = re.findall(r'\d+', element)  # Find all sequences of digits
            for number in digits:
                if len(number) >= 5 and number == element:
                    return element
        return None

    def test_split_areas(self, words, areas):
        split_words = []
        split_areas = []

        # Define a regular expression pattern to split words at spaces, colons, and semicolons
        split_pattern = r'[ \t:;]+'

        for word, area in zip(words, areas):
            # Split the word using the regular expression pattern
            word_split = re.split(split_pattern, word)
            split_words.extend(word_split)

            # Distribute the area proportionally to the number of resulting words
            area_per_word = area / len(word_split)
            split_areas.extend([area_per_word] * len(word_split))

        return split_words, split_areas

    def correct_errors(self, database_words, output_array, areas):
        corrected_words = []
        corrected_areas = []  # Initialize an empty list to store corrected areas
        similarity_threshold = 0.75

        for i, ocr_output in enumerate(output_array):
            selected_word = None
            max_similarity = 0
            for j, word in enumerate(database_words):
                sim = difflib.SequenceMatcher(None, ocr_output, word).ratio()
                n = len(word) if isinstance(word, str) else 0  # Calculate length if it's a string
                if n != 0:
                    similarity_threshold = ((n - 1) / n)

                if sim >= similarity_threshold and sim > max_similarity:
                    selected_word = word
                    max_similarity = sim
            if selected_word:
                corrected_words.append(selected_word)
                corrected_areas.append(areas[i])  # Add the corresponding area
            else:
                corrected_words.append("No suitable match found.")
        filtered_array = [item for item in corrected_words if item != "No suitable match found."]
        return filtered_array, corrected_areas

    def calculate_center_point(self, rectangle):
        x, y, w, h = rectangle
        center_x = x + w // 2
        center_y = y + h // 2
        return center_x, center_y

    def detect_text_in_image(self, image_path, json_file_path):
        img = cv2.imread(image_path)
        reader = easyocr.Reader(['en'])

        result = reader.readtext(img)
        areas = self.calculate_areas(result)

        detected_text = [text[1] for text in result]
        print("OCR Detected Text:", detected_text)

        fin = [word.lower() for word in detected_text]

        output_array, areas2 = self.test_split_areas(fin, areas)

        areas1 = areas
        words = fin + output_array

        all_words = []
        for sub_list in output_array:
            all_words.extend(sub_list)

        correct_words = []
        excel_file_name = "Database2.csv"
        df = pd.read_csv(excel_file_name)
        database = []
        for col in df.columns:
            database.append(df[col].values.tolist())

        correct_words = []

        # Convert non-string elements to strings before adding them to the list
        for word in database[0] + database[1] + database[2] + database[3] + database[4]:

            if isinstance(word, str):
                correct_words.append(word)
            elif isinstance(word, (int, float)):
                correct_words.append(str(word))

        filtered_array, filtered_areas = self.correct_errors(correct_words, output_array, areas2)
        print("Desired Information:", filtered_array)
        i = 0
        while i < len(filtered_array):
            if filtered_array[i] == "nan":
                filtered_array.pop(i)  # Remove the element at index i
                filtered_areas.pop(i)

            else:
                i += 1  # Increment the index only if no removal was done

        all = []
        all_areas = []
        all = filtered_array + fin + output_array
        all_areas = filtered_areas + areas1 + areas2

        center_points = {}
        for detection in result:
            points = detection[0]
            pts = []
            for point in points:
                pts.append([int(x) for x in point])
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            center_x, center_y = self.calculate_center_point(cv2.boundingRect(np.array(pts, dtype=np.int32)))
            text = detection[1]
            center_points[text] = (center_x, center_y)

#         cv2_imshow(img)
        plt.imshow(img)
        plt.show()

        excel_file_name = "Database2.csv"
        df = pd.read_csv(excel_file_name)
        database = []
        for col in df.columns:
            database.append(df[col].values.tolist())

        database[0] = sorted(database[0], key=self.custom_sort)
        database[1] = sorted(database[1], key=self.custom_sort)
        database[2] = sorted(database[2], key=self.custom_sort)
        database[3] = sorted(database[3], key=self.custom_sort)
        database[4] = sorted(database[4], key=self.custom_sort)

        country = "none"
        country = self.find_selected_country(filtered_array, filtered_areas, database)

        fruit = "n/a"
        fruit = self.find_selected_fruit(all, all_areas, database)

        date = []
        date2 = "n/a"
        combined_date = "n/a"
        i = 0
        while i < len(output_array):
            if output_array[i] in database[4]:
                date.insert(0, output_array[i])
                if i > 0 and output_array[i - 1].isdigit():
                    date.insert(0, output_array[i - 1])
            i += 1

        if len(date) >= 2 and not date[-1].isdigit() and date[-2].isdigit():
            combined_date = date[-2] + ' ' + date[-1]
            date = date[:-2]
            date.append(combined_date)

        date_pattern = r'\d+/\d+/\d+'
        # Initialize a list to store detected dates
        all_dates_found = []
        for element in output_array:
            # Find all matching dates in the current element
            dates_found = re.findall(date_pattern, element)
            all_dates_found.extend(dates_found)
        for date1 in all_dates_found:
            date2 = date1

        if (combined_date == "n/a"):
            combined_date = date2

        weight="n/a"

        weight = self.extract_weight(output_array)

        company = "n/a"
        for i in (filtered_array):
            company_found = self.binary_search(i, database[3])
            if company_found:
                company = i

        combined_string =self.extract_large_numbers(output_array)

        fina = ["country: " + str(country), "fruit: " + str(fruit), "weight: " + str(weight),
                "productcode: " + str(combined_string), "best before: " + str(combined_date),
                "company: " + str(company)]

        expdata = []

        for element in fina:
            parts = element.split(":")
            if len(parts) > 1:
                expdata.extend([f"{parts[0]}:", parts[1]])
            else:
                expdata.append(parts[0])

        cleaned_array = [item.strip() for item in expdata]
        print("Generated Output:", cleaned_array)

        cleaned_array = self.select_and_combine_options(cleaned_array)
        print("Selected options:")
        print(cleaned_array)

        with open(json_file_path, 'r') as file:
            json_data = json.load(file)

        json_string = json.dumps(json_data, indent=4)
        word_array = json_string.split()
        formatted_array = self.clean_and_format_array(word_array)

        elemente = 0
        expjson = [item.lower() for item in formatted_array]
        expjson = self.aml(expjson)

        tags = json_data["ExpectedData"]["Tags"]
        tags_list = tags.split(',')
        tags_list = [item.lower() for item in tags_list]
        expjson = expjson + tags_list
        print("Ground Truth:", expjson)
        a = []

        for i in expjson:
            if i in cleaned_array:
                elemente += 1
                a.append(i)

        print(elemente, "common elements")
        a = 0
        n = len(cleaned_array)
        print(((elemente) / ((n / 2) - a)) * 100, "% common elements percentage")

    def process_image(self, image_path, json_file_path):
      self.detect_text_in_image(image_path, json_file_path)


   

# ... Your existing code ...

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Text Recognition from Image with Customized Paths")

    # Add arguments for image_path and json_file_path
    parser.add_argument("--image", type=str, help="Path to the input image")
    parser.add_argument("--json", type=str, help="Path to the JSON file")

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.image is None or args.json is None:
        print("Please provide paths for both the image and JSON file using --image and --json arguments.")
    else:
        text_recognition = TextRecognition()
        image_path = args.image
        json_file_path = args.json
        text_recognition.process_image(image_path, json_file_path)
