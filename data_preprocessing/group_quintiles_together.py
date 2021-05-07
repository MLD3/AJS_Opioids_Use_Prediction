from features import time_dep_features, time_invar_features, recency_features

### all features are of the form "name_value_<actual value or bucket>"
### thus, two features are variations of the same property if they are equal up to the final "_value"

name_terminator = "_value"

### This is the correct order 
feature_names = time_dep_features + time_invar_features + recency_features

### list_of_related_features will be a list of lists, where each list within the list will include the
### indices of all features with names matching up to the final "_value"
def groupRelatedFeaturesIntoLists(feature_names):
    list_of_related_features = []

    group_index = 0

    def featurePrefix(feature_name):
        name_end_index = feature_name.rfind('_value')
        return feature_name[0:name_end_index]

    while group_index < len(feature_names):
        related_features = [group_index]
    
        while group_index < len(feature_names) - 1 and featurePrefix(feature_names[group_index]) == featurePrefix(feature_names[group_index+1]):
            group_index += 1
            related_features.append(group_index)

        group_index += 1
        list_of_related_features.append(related_features)

    return list_of_related_features

list_of_related_features = groupRelatedFeaturesIntoLists(feature_names)

print(list_of_related_features)
print(len(feature_names))


