from vico.crawl.download_spider import DownloadSpider, Link


class ParfumdreamsSpider(DownloadSpider):
    name = 'ParfumdreamsSpider'

    def __init__(self, **kwargs):
        super().__init__(domain='parfumdreams', **kwargs)

    def get_links(self, response):
        for link in response.css('a.item-link'):
            url = link.css('::attr(href)').extract_first()
            yield Link(url=url, tag=link)


