import axios from 'axios';
import * as cheerio from 'cheerio';

const webScraper = async (url) => {
  try {
    const response = await axios.get(url);
    console.log(response.data);
    const $ = cheerio.load(response.data);
    const externalLinks = [];
    const internalLinks = [];
    const pageBody = $('body').text();
    const pageHead = $('head').text();
    $('a').each((i, elem) => {
      if (link==='/') return;
      // Filter out links that are not relevant
      if (link.includes('mailto:') || link.includes('tel:') || link.includes('#')) return;
      // Extract the link and log it
      if(link.startsWith('http')||link.startsWith('https')) {
        externalLinks.push(link);
      }
      else {
        internalLinks.push(link);
      }
      return {head:pageHead, body: pageBody, externalLinks, internalLinks};
    });
  } catch (error) {
    console.error('Error fetching the URL:', error);
  }
}
webScraper('http://localhost:5173/').then((data) => {
  console.log(data?.head);
  console.log(data?.body);
  console.log('External Links:', data.externalLinks);
  console.log('Internal Links:', data.internalLinks);
})