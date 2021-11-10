import unittest
import requests

from src import collect

class CollectorCheck(unittest.TestCase):

    def test_header_params(self):
        collector = collect.APICollector()
        self.assertDictEqual(collector.params, {})
        self.assertDictEqual(collector.headers, {})

    def test_attributes(self):
        collector = collect.APICollector()
        self.assertIsInstance(collector.sess, requests.Session)

    def test_get_request(self):
        collector = collect.APICollector()
        URL = 'https://httpbin.org/status/{}'

        response = collector.get(URL.format(200))
        response2 = collector.get(URL.format(439))
        response3 = collector.get(URL.format(501))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response2.status_code, 439)
        self.assertEqual(response3.status_code, 501)


if __name__ == '__main__':
    unittest.main()
