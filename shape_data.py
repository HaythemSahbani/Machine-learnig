import numpy as np
from Segmentation import Segmentation
import glob


def load_files(source_dir, data_type):
    """
    This function loads files in the source_directory, with the type as extension
    :param source_dir: source directory
    :param type: extension of the files
    :return: list
    """
    file_list = glob.glob(source_dir + '/*.'+data_type)
    print("number of files = ", len(file_list))
    for file_path in file_list:
        yield np.loadtxt(file_path, delimiter=" ")


def save_data(result_file, data):
    """
    saves multiple numpy arrays in one file.
    :param result_file:
    :param data: takes a generator of numpy array
    :return:
    """
    segment = Segmentation()
    # for element in data:
    try:
        while True:
            element = next(data)  # generator
            element = segment.format_data(segment.segmentMaryam(element))
            with open(result_file, 'a') as f_handle:
                np.savetxt(f_handle, element,  delimiter=" ", footer='####################################'
                                                                     '######################################'
                                                                     '#######################################'
                                                                     '############')
    except StopIteration:
        pass


def load_data(data_file):
    """
    loads all the data in one dictionary.
    The dictionary takes the subject number as key and the array as value
    :param data_file:
    :return: dictionary
    """
    dictionary = {}
    data_set = np.loadtxt(data_file, delimiter=" ")
    subject_number = 0
    index = np.where(data_set[:, 0] == 0)
    try:
        for i in range(0, np.shape(index)[1]):
            subject_number += 1
            subject = 'subject ' + str(subject_number)
            dictionary.update({subject: data_set[index[0][i]: index[0][i+1]-1]})
    except:
        dictionary[subject] = data_set[index[0][i]:]
    return dictionary


def cross_validation(dictionary):
    """
    parses the data to enable cross validation.
    This function takes as input the result of the load_data function
    :param dictionary:
    :return:
    """
    result_dictionary = dict([('test data', np.array([])), ('training data', np.array([])), ('test subject', "")])
    for i in range(len(dictionary)):
        data_list = np.zeros(shape=(0, 22))
        for j in range(len(dictionary)):
            if j != i:
                data_list = np.append(data_list, dictionary.values()[j], axis=0)

        result_dictionary.update(({'training data': np.array(data_list)}))
        result_dictionary.update(({'test data': np.array(dictionary.values()[i])}))
        result_dictionary.update({'test subject': dictionary.keys()[i]})

        yield result_dictionary


def merge_data_set(protocol_result_file="PAMAP2_Dataset/Protocol",
                   optional_result_file="PAMAP2_Dataset/Optional",
                   total_result_file="total data result file.dat"):
    """
    Merge the protocol and optional files by subject into one resulting file.
    :param protocol_result_file:
    :param optional_result_file:
    :param total_result_file:
    :return:
    """
    protocol_dic = load_data(protocol_result_file)
    optional_dic = load_data(optional_result_file)


    protocol_dic["subject 1"] = np.append(protocol_dic["subject 1"], optional_dic["subject 1"], axis=0)
    protocol_dic["subject 5"] = np.append(protocol_dic["subject 5"], optional_dic["subject 2"], axis=0)
    protocol_dic["subject 6"] = np.append(protocol_dic["subject 6"], optional_dic["subject 3"], axis=0)
    protocol_dic["subject 8"] = np.append(protocol_dic["subject 8"], optional_dic["subject 4"], axis=0)
    protocol_dic["subject 9"] = np.append(protocol_dic["subject 9"], optional_dic["subject 5"], axis=0)
    for i in range(np.shape(protocol_dic["subject 9"])[0]):
        protocol_dic["subject 9"][i, 0] = i
    for i in range(np.shape(protocol_dic["subject 8"])[0]):
        protocol_dic["subject 8"][i, 0] = i
    for i in range(np.shape(protocol_dic["subject 6"])[0]):
        protocol_dic["subject 6"][i, 0] = i
    for i in range(np.shape(protocol_dic["subject 5"])[0]):
        protocol_dic["subject 5"][i, 0] = i
    for i in range(np.shape(protocol_dic["subject 1"])[0]):
        protocol_dic["subject 1"][i, 0] = i
    for subject_number in range(1, len(protocol_dic.keys()) + 1):
        print("merging subject number:  ", subject_number)
        with open(total_result_file, 'a') as f_handle:
            np.savetxt(f_handle, protocol_dic["subject " + str(subject_number)],  delimiter=" ", footer='####################################'
                                                                            '######################################'
                                                                            '#######################################'
                                                                           '############')
    print " Data set merged and saved in %s ",  total_result_file


def main():

    # The folder containing the Optional data set
    optional_data = "PAMAP2_Dataset/Optional"
    # The folder containing the Protocol data set
    protocol_data = "PAMAP2_Dataset/Protocol"

    # Load the data from the folders
    optional_data_files = load_files(optional_data, "dat")
    protocol_data_files = load_files(protocol_data, "dat")

    # Resulting files
    protocol_result_file = " protocol result file.dat"
    optional_result_file = "optional result file.dat"
    total_result_file = "total data result file.dat"

    save_data(protocol_result_file, protocol_data_files)
    save_data(optional_result_file, optional_data_files)

    print "Data set processed and saved in these files: %s and %s", protocol_result_file, optional_result_file





    print "Data set processed and saved"

if __name__ == "__main__":
    main()
