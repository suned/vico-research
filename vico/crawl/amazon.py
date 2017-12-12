from vico.crawl.download_spider import DownloadSpider, Link


class AmazonSpider(DownloadSpider):
    name = 'AmazonSpider'

    def __init__(self, **kwargs):
        super().__init__(domain='amazon', **kwargs)

    def get_links(self, response):
        for customers_also_bought in response.css('.a-carousel > li > div > a'):
            url = customers_also_bought.css(
                '.a-link-normal::attr(href)'
            ).extract_first()
            yield Link(url=url, tag=customers_also_bought)
