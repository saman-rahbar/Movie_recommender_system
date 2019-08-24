
import surprise as surp
import pandas as pd
class collaborative_filtering:
    def __init__(self, df_rating, test_ratio=None, df_id_name_table=None, rating_scale=(1, 5))
        """
        Initialize collaborative filtering data class
        :param df_rating: pandas dataframe containg columns: 'userID', 'itemID', 'rating' in correct order
        :param df_id_name_table: table to convert item_ID to readble item name (e.g. Movie titles)
                                dataframe containing columns: 'itemID' and 'itemName' in correct order
        
        :return: None
        
        """
        reader = surp.reader(rating_scale=rating_scale)
        rating_data = surp.Dataset.load_from_df(df_rating, reader)
        self.tainset = rating_data.build_full_trainset()

        if test_ratio is not None:
            self.trainset, self.testset = surp.model_selection.train_test_split(data=rating_data, test_size= test_ratio)
        else:
            self.trainset = rating_data.build_full_trainset()

        # self._dict_id_to_name: id_1: [name1_1, name1_2...], id_1: [name2_1, name2_2...]
        # self._dict_name_to_id: name1: [id1_1, id1_2...], name2: [id2_1, id2_2...]
        if df_id_name_table is not None:
            self._dict_id_to_name = df_id_name_table.groupby('itemID')['itemName'].apply(lambda x: x.tolist()).to_dict()
            self._dict_name_to_id = df_id_name_table.groupby('itemName')['itemID'].apply(lambda x: x.tolist()).to_dict()

    def convert_name_to_id(self, item_name):
        """
        Converts item name to item id
        :param item_name: item name
        :return: item id if single id is not found, None if nothing or multiple ids are found
        """
        if item_name not in self._dict_name_to_id or len(self._dict_name_to_id[item_name]) > 1:
            return None
        return self._dict_name_to_id[item_name][0]

    def convert_id_to_name(self, item_id):
        """
        Converts item id to item name
        :param item_id: item id
        :return: item name if single name is found, None if nothing or multiple ids are found
        """
        if item_id not in self.__dict_id_to_name or len(self.__dict_id_to_name[item_id]) > 1:
                return None
        return self.__dict_id_to_name[item_id][0]
