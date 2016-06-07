# scrapy runspider scrape_imsdb.py

import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import Selector

class IMSDBSpider(CrawlSpider):
    name = "imsdb"
    allowed_domains = ["imsdb.com"]
    start_urls = ["http://www.imsdb.com/TV/Futurama.html"]

    rules = (
        Rule(LinkExtractor(allow="TV Transcripts\/Futurama")),
        Rule(LinkExtractor(allow="\/transcripts"), callback="get_script")
    )

    def get_script(self, response):
        print response.url
        # print Selector(response=response).xpath("//pre").extract()[0]
        with open("rawdata/" + response.url.split("/")[-1].split(".")[0] + ".txt", "w") as f:
            f.write(Selector(response=response).xpath("//pre").extract()[0].encode('ascii', 'ignore'))
