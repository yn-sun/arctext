from path import convertTools as ct

if __name__ == '__main__':
    print("Converting NASBench to arcText dataframe...")
    ct.convert_NASBench_to_arcTextDf(record_size=20000)

    print("Generate onehot dictionary according to arcText dataframe...")
    max_vector_size = ct.get_onehot_dict_from_arcTextDf()
    print("max vector size is %d" %max_vector_size)

    print("Generate dataset for running and evaluating model ...")
    ct.generate_dataset_from_arcTextDf_and_onehoDict()

    print("Complete!")
