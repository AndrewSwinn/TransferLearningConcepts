import os.path
import sqlite3
import socket
import numpy as np
np.seterr(divide='ignore')

from torch.utils.data import Dataset

if socket.gethostname() == 'LTSSL-sKTPpP5Xl':
    data_dir = 'C:\\Users\\ams90\\PycharmProjects\\ConceptsBirds\\data'
elif socket.gethostname() == 'LAPTOP-NA88OLS1':
    data_dir = 'D:\\data\\caltecBirds\\CUB_200_2011'
else:
    data_dir = '/home/bwc/ams90/datasets/caltecBirds/CUB_200_2011'



drop_statements = [ "drop table if exists classes;",
                    "drop table if exists concepts;",
                    "drop table if exists attributes;",
                    "drop table if exists images;",
                    "drop table if exists image_attributes;",
                    "drop table if exists class_attribute_probabilities;",
                    "drop table if exists nb_class_counts;",
                    "drop table if exists nb_attribute_counts;"
                    ]

create_statements = ["create table if not exists classes (class_id integer primary key, class_name text not null);",
                     "create table if not exists concepts (concept_id integer primary key, concept_name text not null);",
                     "create table if not exists attributes (attribute_id integer not null, concept_id integer not null, value_id integer not null, value text not null);",
                     "create table if not exists images (image_id integer primary key, filename text not null, class_id integer, trainset integer, box_x integer, box_y integer, box_w integer, box_h integer);",
                     "create table if not exists image_attributes (image_id integer not null, attribute_id integer not null, present integer not null, certainty integer not null);",
                     "create index if not exists image_attributes_u1 on image_attributes (image_id, attribute_id);",
                     "create table if not exists class_attribute_probabilities (class_id integer not null, attribute_id integer not null, probability float, naive_train_probability float, naive_test_probability float);",
                     "create index if not exists class_attribute_probabilities_n1 on class_attribute_probabilities (attribute_id);",
                     """create table if not exists nb_class_counts    
                                         (class_id integer, trainset not null, class_count integer not null, class_prior float not null,
                                         primary key (class_id, trainset));""",
                     """create table if not exists nb_attribute_counts 
                                         (class_id integer, attribute_id integer, trainset, attribute_count integer , log_likelihood float, log_likelihood_conjugate float,
                                          primary key(class_id, attribute_id, trainset) );"""
                     ]

def get_concept_id(concept):
    def sel_id():
        get_sql = 'select concept_id from concepts where concept_name = ?'
        return concept_cursor.execute(get_sql, (concept,)).fetchone()

    concept_cursor = conn.cursor()
    concept_record = sel_id()
    if concept_record is None:
        ins_sql = 'insert into concepts (concept_name) values (?)'
        concept_cursor.execute(ins_sql, (concept,))
        concept_record = sel_id()
    concept_cursor.close()

    return concept_record[0]

insert_class                         = 'insert into classes (class_id, class_name) values (?,?)'
insert_attribute                     = 'insert into attributes (attribute_id, concept_id, value_id, value) values (?,?,?,?)'
insert_image                         = 'insert into images (image_id, filename, class_id, trainset, box_x, box_y, box_w, box_h) values (?,?, ?, ?, ?, ?, ?, ?)'
insert_image_attributes              = 'insert into image_attributes (image_id, attribute_id, present, certainty) values (?, ?, ?, ?)'
insert_class_attribute_probabilities = 'insert into class_attribute_probabilities (class_id, attribute_id, probability, naive_train_probability, naive_test_probability) values (?, ?, ?, ?, ?)'


if __name__ == '__main__':

    conn = sqlite3.connect(database=os.path.join(data_dir, 'birds.db'))
    cursor = conn.cursor()
    conn.create_function("log", 1, np.log)
    conn.create_function("exp", 1, np.exp)

    [cursor.execute(statement) for statement in drop_statements]
    [cursor.execute(statement) for statement in create_statements]

    class_file = open(os.path.join(data_dir, 'classes.txt'))
    for class_record in class_file:
        class_id, description = class_record.split(sep=" ")
        description = description.replace('\n','')
        cursor.execute(insert_class,(class_id, description) )

    previous_concept_id = -1
    attribute_file = open(os.path.join(data_dir, 'attributes.txt'))
    for attribute_record in attribute_file:
        attribute_id, description = attribute_record.split(sep=" ")
        concept, value = description.replace('\n','').split(sep="::")
        concept_id     = get_concept_id(concept)
        if concept_id != previous_concept_id:
            #cursor.execute(insert_attribute, (0, concept_id, 0, 'Unknown'))
            previous_concept_id, value_id = concept_id, 1
        cursor.execute(insert_attribute, (attribute_id, concept_id, value_id, value))
        value_id += 1

    image_file         = open(os.path.join(data_dir, 'images.txt'))
    image_class_labels = open(os.path.join(data_dir, 'image_class_labels.txt'))
    train_test_split   = open(os.path.join(data_dir, 'train_test_split.txt'))
    bounding_boxes     = open(os.path.join(data_dir, 'bounding_boxes.txt'))
    for image_record in image_file:
        image_id, filename = image_record.replace('\n','').split(sep=" ")
        class_image_id, class_id = image_class_labels.readline().replace('\n','').split(sep=" ")
        test_image_id, trainset  = train_test_split.readline().replace('\n','').split(sep=" ")
        box_image_id, box_x, box_y, box_w, box_h = bounding_boxes.readline().replace('\n','').split(sep=" ")
        if image_id != class_image_id or test_image_id != image_id or box_image_id != image_id:
            raise Exception('File image_ids are not aligned')
        cursor.execute(insert_image, (image_id, filename, class_id, trainset, box_x, box_y, box_w, box_h))

    image_attribute_labels = open(os.path.join(data_dir, 'attributes', 'image_attribute_labels.txt'))
    for image_attribute_label_record in image_attribute_labels:
        record_split = image_attribute_label_record.replace('\n','').split(sep=" ")
        [image_id, attribute_id, present, certainty] = record_split[0:4]
        cursor.execute(insert_image_attributes, (image_id, attribute_id, present, certainty))

    for dataset in [0,1]:

        (dataset_count,) = cursor.execute("select count(image_id) from images where trainset = ?", (dataset,)).fetchone()

        cursor.execute("""insert into nb_class_counts
                                    (class_id, trainset, class_count, class_prior)
                                    select class_id, trainset, count(*), count(*) / ?
                                    from   images
                                    where  trainset = ?
                                    group by class_id,
                                             trainset""", (float(dataset_count), dataset))

    #set smoothing constant - used to prevent probabilities being 0, or 1
    alpha = 1
    cursor.execute("""insert into nb_attribute_counts
                      (class_id, attribute_id, trainset, attribute_count, log_likelihood, log_likelihood_conjugate)
                      select   i.class_id, 
                               i.trainset, 
                               ia.attribute_id, 
                               count(ia.attribute_id),
                               log((? +                   sum(ia.present)) / ((2.0 * ?) +  ncc.class_count)) ,
                               log((? + ncc.class_count - sum(ia.present)) / ((2.0 * ?) +  ncc.class_count))
                      from     images i,
                               image_attributes ia,
                               nb_class_counts  ncc
                      where    i.image_id = ia.image_id
                      and      ncc.class_id = i.class_id
                      and      ncc.trainset = i.trainset
                      group by i.class_id, 
                               i.trainset, 
                               ia.attribute_id""", (alpha, alpha, alpha, alpha))
                      




    naive_probabilities = np.zeros(shape=(2, 200, 312))

    for trainset in [0,1]:

        attribute_count = [record for record in cursor.execute(
                                                            """select i.class_id, 
                                                                      ia.attribute_id, 
                                                                      log(avg(ia.present)) 
                                                               from images i,
                                                                    image_attributes ia 
                                                               where i.trainset = ?
                                                               and   i.image_id = ia.image_id
                                                               group by i.class_id, 
                                                                        ia.attribute_id""", (trainset,)).fetchall()]

        for (class_id, attribute_id, probability) in attribute_count:
            naive_probabilities[trainset, class_id - 1, attribute_id - 1] = probability

    #create table based on the class_attribute_labels_continuous.txt datafile
    class_attributes_probabilities = open(os.path.join(data_dir, 'attributes', 'class_attribute_labels_continuous.txt'))
    for i, attributes_probabilities in enumerate(class_attributes_probabilities):
        class_id, attribute_id = i+1, 1
        attributes_probabilities_list = attributes_probabilities.replace('\n','').split(sep=" ")
        for j, probability in enumerate(attributes_probabilities_list):
            probability = np.log(float(probability)/100.)
            attribute_id = j+1
            cursor.execute(insert_class_attribute_probabilities, (class_id, attribute_id, probability,
                                                                  naive_probabilities[1, class_id - 1, attribute_id - 1], naive_probabilities[0, class_id - 1, attribute_id - 1]))

    conn.commit()

