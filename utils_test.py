import collections
import requests
import unittest
import unittest.mock as mock
import utils

Response = collections.namedtuple('Response', ['status_code', 'content'])


class UtilsTest(unittest.TestCase):

    def test_web_scraping_success(self):
        fake_response = Response(
            200, b'{volume: 100, price: 1,325.17, symbol: QQQ}')
        with mock.patch.object(requests, 'get', return_value=fake_response):
            price = utils.web_scraping('fake_url', 'price')
        self.assertEqual(price, '1325.17')

    def test_web_scraping_network_error(self):
        fake_response = Response(404, b'')
        with mock.patch.object(requests, 'get', return_value=fake_response) as fake_get:
            with self.assertRaises(utils.NetworkError):
                utils.web_scraping('fake_url', 'price')
        self.assertEqual(fake_get.call_count, 3)


if __name__ == '__main__':
    unittest.main()
