#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    i, total = 0, len(predictions)

    while i < total:
        age, net_worth = ages[i], net_worths[i]
        error = abs(predictions[i] - net_worths[i])
        tp = ( age, net_worth, error )
        cleaned_data.append(tp)
        i += 1

    cleaned_data = sorted(cleaned_data, key=lambda tup: tup[2])
    ten_percent = total * .90
    
    i = ten_percent
    while i < total:
        cleaned_data.pop()
        i += 1

    return cleaned_data
