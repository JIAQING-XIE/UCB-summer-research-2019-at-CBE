###Data preprocessing

nor_x = Normalizer()                         
x_train =nor_x.fit_transform(x_train)
x_test = nor_x.fit_transform(x_test)

ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test  = ss_x.fit_transform(x_test)

mm_x = MinMaxScaler()
x_train = mm_x.fit_transform(x_train)
x_test  = mm_x.fit_transform(x_test)




            
