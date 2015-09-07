import numpy as np


class Segmentation:

    def __init__(self):
        pass

    @staticmethod
    def activityToMet (activityID):
        if activityID == 1 or activityID == 9:
            return 1
        elif activityID == 2 or activityID == 3:
            return 1.8
        elif activityID == 4 or activityID == 16:
            return 3.5
        elif activityID == 5:
            return 7.5
        elif activityID == 6:
            return 4
        elif activityID == 7:
            return 5
        elif activityID == 10:
            return 1.5
        elif activityID == 11 or activityID == 18:
            return 2
        elif activityID == 12:
            return 8
        elif activityID == 13 or activityID == 19:
            return 3
        elif activityID == 17:
            return 2.3
        elif activityID == 20:
            return 7
        elif activityID == 24:
            return 9
        else:
            return 0

    @staticmethod
    def metToLevel(met):
        if met < 3:
            # return "light"
            return 1
        elif met < 6:
            # return "moderate"
            return 2
        else:
            # return "vigorous"
            return 3

    @staticmethod
    def metToLevelBinary(met):
        if met < 6:
            #return "light"
            return 0
        else:
            #return "vigorous"
            return 1

    @staticmethod
    def energy_extraction(data):
        """
        :param data:
        :return:
        Energy is the sum of the squared discrete FFT
        component magnitudes of the signal. The sum was divided
        by the window length for normalization.
        """
        # remove nan values
        data = data[~np.isnan(data)]
        data = np.fft.fft(data)
        data = np.mean(abs(np.power(data, 2)))
        return data

    @staticmethod
    def heart_rate_extraction(data, maxHeartRate, minHeartReate):
        """
        :param data: numpy array
        :param maxHeartRate:
        :param minHeartReate:
        :return:
        """
        # remove nan values
        data = data[~np.isnan(data)]
        # normalized heart rate: max - x/(max-min)
        # #
        return (maxHeartRate - data.mean())/(maxHeartRate - minHeartReate)

    @staticmethod
    def stdDev(data):
        data = [~np.isnan(data)] #deleted a
        return data.std()

    @staticmethod
    def segmentNp(data): #data: np.array
        #nrOfCols = np.shape(data)[1]
        nrOfCols = 12 #ouput will have 10 Features (+ id col, + label col)
        result = np.array([np.zeros(nrOfCols)])

        '''
        Dataformate:
        0: Id/timestamp
        1: activity label
        2: heart rate
        
        3: IMU Hand temperature
        4: IMU Hand 3D accerleration 1
        5: IMU Hand 3D accerleration 2
        6: IMU Hand 3D accerleration 3
        
        21: IMU chest 3D accerleration 1
        22: IMU chest 3D accerleration 2
        23: IMU chest 3D accerleration 3
        
        38: IMU ankle 3D accerleration 1
        39: IMU ankle 3D accerleration 2
        40: IMU ankle 3D accerleration 3
        '''
        for i in range(0, len(data)//512): # ..//.. -> Full number division when future imported
            line = np.zeros(nrOfCols)
            line[0] = i
            line[1] = data[i*512+256][1] #MET Label
            #line[2] = np.nanmean(data[i*512:i*512+512,2])
            line[2] = self.heart_rate_extraction(data[i*512:i*512+512,2], 80, 160) # heart Rate
            '''
            #fft
            line[3] = self.energy_extraction(data[i*512:i*512+512,4])
            line[4] = self.energy_extraction(data[i*512:i*512+512,5])
            line[5] = self.energy_extraction(data[i*512:i*512+512,6])
            line[6] = self.energy_extraction(data[i*512:i*512+512,21])
            line[7] = self.energy_extraction(data[i*512:i*512+512,22])
            line[8] = self.energy_extraction(data[i*512:i*512+512,23])
            line[9] = self.energy_extraction(data[i*512:i*512+512,38])
            line[10] = self.energy_extraction(data[i*512:i*512+512,39])
            line[11] = self.energy_extraction(data[i*512:i*512+512,40])
            '''
            #std
            line[3] = np.nanstd(data[i*512:i*512+512,4])
            line[4] = np.nanstd(data[i*512:i*512+512,5])
            line[5] = np.nanstd(data[i*512:i*512+512,6])
            line[6] = np.nanstd(data[i*512:i*512+512,21])
            line[7] = np.nanstd(data[i*512:i*512+512,22])
            line[8] = np.nanstd(data[i*512:i*512+512,23])
            line[9] = np.nanstd(data[i*512:i*512+512,38])
            line[10] = np.nanstd(data[i*512:i*512+512,39])
            line[11] = np.nanstd(data[i*512:i*512+512,40])

            result = np.append(result, np.array([line]), axis = 0)

        return result[1:] #remove first line because we set it to 0 in the beginning

    @staticmethod
    def segmentMaryam(data_set):
        """
        :param data_set:
        :return:
        """
        # take the 0 index
        zeros_index = np.where(data_set[:, 1] == 0)
        # delete the 0 rows
        data_set = np.delete(data_set, zeros_index, 0)
        finalDataSet=[]
        toBeDeleted=[]
        beginning = 0
        notUsable = 0

        dataSetLength = len(data_set)
        for i in range(1, dataSetLength):
                if (data_set[i][1] != data_set[i-1][1]) and i>0:

                    for m in range(beginning, beginning+1024):
                        toBeDeleted.append(m)

                    notUsable = ((i - beginning - 1024) % 512)

                    for m in range(i-1024-notUsable, i):
                        toBeDeleted.append(m)

                    beginning = i 
                    notUsable = 0

        for m in range(beginning, beginning+1024):
            toBeDeleted.append(m)
            notUsable = ((dataSetLength - beginning - 1024)%512)
            for m in range(dataSetLength-notUsable-1024, dataSetLength):
                toBeDeleted.append(m)

        #for not_usable_rows in toBeDeleted:
        data_set = np.delete(data_set, toBeDeleted, 0)

        return data_set

    def format_data(self, data):
        """
        extracts the features from the input data and saves it in a numpy array
        the feature extracted are:
                energy
                standard deviation
                normalised heart rate
        :param data:
        :return:
        """
        result_array = np.zeros(shape=(0, 22))  # we have 21 result columns
        maxHR = np.max(data[:, 2][~np.isnan(data[:, 2])])
        minHR = np.min(data[:, 2][~np.isnan(data[:, 2])])
        for i in range(0, len(data), 512):
            line = np.zeros(shape=(1, 22))
            line[:, 0] = i / 512
            # activity level: light, moderate or vigorous
            line[:, 1] = self.metToLevel(self.activityToMet(data[i][1]))
            # activity MET
            line[:, 2] = self.activityToMet(data[i][1])
            # heart Rate extraction
            line[:, 3] = self.heart_rate_extraction(data[i:i+512, 2], maxHeartRate=maxHR, minHeartReate=minHR)

            """
            Energy extraction
            """
            # IMU Hand 3D acceleration  columns 4, 5, 6, scale: 16g, resolution: 13-bit
            line[:, 4] = self.energy_extraction(data[i:i+512, 4])
            line[:, 5] = self.energy_extraction(data[i:i+512, 5])
            line[:, 6] = self.energy_extraction(data[i:i+512, 6])
            # IMU Chest 3D acceleration  columns 21, 22, 23, scale: 16g, resolution: 13-bit
            line[:, 7] = self.energy_extraction(data[i:i+512, 21])
            line[:, 8] = self.energy_extraction(data[i:i+512, 22])
            line[:, 9] = self.energy_extraction(data[i:i+512, 23])
            # IMU ankle 3D acceleration columns 38, 39, 40, scale: 16g, resolution: 13-bit
            line[:, 10] = self.energy_extraction(data[i:i+512, 38])
            line[:, 11] = self.energy_extraction(data[i:i+512, 39])
            line[:, 12] = self.energy_extraction(data[i:i+512, 40])

            """
            standard deviation extraction
            """
            # IMU Hand 3D acceleration  columns 4, 5, 6, scale: 16g, resolution: 13-bit
            line[:, 13] = np.nanstd(data[i:i+512, 4])
            line[:, 14] = np.nanstd(data[i:i+512, 5])
            line[:, 15] = np.nanstd(data[i:i+512, 6])
            # IMU Chest 3D acceleration  columns 21, 22, 23, scale: 16g, resolution: 13-bit
            line[:, 16] = np.nanstd(data[i:i+512, 21])
            line[:, 17] = np.nanstd(data[i:i+512, 22])
            line[:, 18] = np.nanstd(data[i:i+512, 23])
            # IMU ankle 3D acceleration columns 38, 39, 40, scale: 16g, resolution: 13-bit
            line[:, 19] = np.nanstd(data[i:i+512, 38])
            line[:, 20] = np.nanstd(data[i:i+512, 39])
            line[:, 21] = np.nanstd(data[i:i+512, 40])

            result_array = np.append(result_array, line, axis=0)
        # result_array = np.delete(result_array, 0, 0)
        return result_array