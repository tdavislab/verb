from locust import HttpUser, task
import json


class QuickstartUser(HttpUser):
    @task
    def create_post(self):
        headers = {'content-type': 'application/x-www-form-urlencoded; charset=UTF-8', 'Accept-Encoding': 'gzip, defalte', 'HTTP_USER_AGENT': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:84.0) Gecko/20100101 Firefox/84.0',}
        self.client.post("seedwords2",
                         data={
                             'algorithm': 'Algorithm: Linear projection',
                             'concept1_name': 'Gender',
                             'concept2_name': 'Concept2',
                             'equalize': 'man-woman,he-him,she-her',
                             'evalwords': 'engineer, lawyer, receptionist, homemaker',
                             'orth_subspace': 'scientist, doctor, nurse, secretary, maid, dancer, cleaner, advocate, player, banker',
                             'seedwords1': 'he',
                             'seedwords2': 'she',
                             'subspace_method': 'Subspace method: Two means'},
                         headers=headers)

#
# class WebsiteUser(HttpUser):
#     task_set = UserBehavior
# #
#
# class QuickstartUser(HttpUser):
#     wait_time = between(1, 2.5)
#
#     @task
#     def make_request(self):
#         self.client.post("seedwords2", body={
#             'algorithm': 'Algorithm: Linear projection',
#             'concept1_name': 'Gender',
#             'concept2_name': 'Concept2',
#             'equalize': 'man-woman,he-him,she-her',
#             'evalwords': 'engineer, lawyer, receptionist, homemaker',
#             'orth_subspace': 'scientist, doctor, nurse, secretary, maid, dancer, cleaner, advocate, player, banker',
#             'seedwords1': 'he',
#             'seedwords2': 'she',
#             'subspace_method': 'Subspace method: Two means'
#         })
