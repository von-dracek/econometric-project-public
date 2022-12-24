### Ekonometricky projektovy seminar 2022/2023

#file paths have to be adjusted
#nacitame vsetky cesty pre "certificates.csv", resp. pre "recommendations.csv"
files_paths <- Sys.glob(file.path("/srv/beegfs/cluster/home/till/barclays_data", "*.csv"))

#nacitanie dat small
test_x_small = read.csv(files_paths[2])
train_x_small = read.csv(files_paths[4])
train_y_small = read.csv(files_paths[6])

#mergujeme train_small data a CURRENT_ENERGY_RATING priradime poradie
train_small <- train_x_small
train_small$CURRENT_ENERGY_RATING <- ordered(train_y_small$CURRENT_ENERGY_RATING)

#zmenime formaty kategorickych premennych
train_small$BUILT_FORM <- as.factor(train_small$BUILT_FORM)
train_small$IS_EPC_LABEL_BEFORE_2008_INCL <- as.factor(train_small$IS_EPC_LABEL_BEFORE_2008_INCL)
train_small$LOCAL_AUTHORITY_LABEL <- as.factor(train_small$LOCAL_AUTHORITY_LABEL)
train_small$POSTCODE_PROPORTIONS_ARE_RELIABLE_IND <- as.factor(train_small$POSTCODE_PROPORTIONS_ARE_RELIABLE_IND)
train_small$PROPERTY_TYPE <- as.factor(train_small$PROPERTY_TYPE)
test_x_small$BUILT_FORM <- as.factor(test_x_small$BUILT_FORM)
test_x_small$IS_EPC_LABEL_BEFORE_2008_INCL <- as.factor(test_x_small$IS_EPC_LABEL_BEFORE_2008_INCL)
test_x_small$LOCAL_AUTHORITY_LABEL <- as.factor(test_x_small$LOCAL_AUTHORITY_LABEL)
test_x_small$POSTCODE_PROPORTIONS_ARE_RELIABLE_IND <- as.factor(test_x_small$POSTCODE_PROPORTIONS_ARE_RELIABLE_IND)
test_x_small$PROPERTY_TYPE <- as.factor(test_x_small$PROPERTY_TYPE)

library(MASS)
gc()
ord_small <- polr(CURRENT_ENERGY_RATING ~ ., data = train_small)
summary(ord_small)
ord_small_predpoved <- predict(ord_small, test_x_small, type="probs")
write.csv(ord_small_predpoved,"/srv/beegfs/cluster/home/till/output/predicts_small.csv", row.names = FALSE)
