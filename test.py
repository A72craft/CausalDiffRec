import dgl

glist, label_dict = dgl.load_graphs("dataset/review/review_test_data.bin")
print(glist)