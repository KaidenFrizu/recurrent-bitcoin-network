import unittest
import requests

from src import collect

class CollectorCheck(unittest.TestCase):

    def setUp(self):
        global collector
        global URL

        collector = collect.APICollector()
        URL = 'https://httpbin.org/status/{}'

    def test_attributes(self):
        self.assertIsInstance(collector.headers, dict)
        self.assertIsInstance(collector.params, dict)
        self.assertIsInstance(collector.sess, requests.Session)

    def test_header_params(self):
        self.assertDictEqual(collector.params, {})
        self.assertDictEqual(collector.headers, {})

        collector.headers.update({'Content-type':'application/json'})

        self.assertDictEqual(
            collector.headers,
            {'Content-type':'application/json'}
        )

    def test_get_request(self):
        response = collector.get(URL.format(200))
        self.assertEqual(response.status_code, 200)

        response2 = collector.get(URL.format(439))
        self.assertEqual(response2.status_code, 439)

        response3 = collector.get(URL.format(500))
        self.assertEqual(response3.status_code, 500)

        response4 = collector.get(URL.format(501))
        self.assertEqual(response4.status_code, 501)


if __name__ == '__main__':
    unittest.main()
