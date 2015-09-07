def activitoToMET (activityID):
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
        return None
def metToLevel (met):
    if met is None:
        pass
    else:
        if met < 3:
            return 1
        elif met < 6:
            return 2
        else:
            return 3

