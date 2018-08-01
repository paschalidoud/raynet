function getSearchTerm()
{
    var sPageURL = window.location.search.substring(1);
    var sURLVariables = sPageURL.split('&');
    for (var i = 0; i < sURLVariables.length; i++)
    {
        var sParameterName = sURLVariables[i].split('=');
        if (sParameterName[0] == 'q')
        {
            return sParameterName[1];
        }
    }
}

$(document).ready(function() {

    var search_term = getSearchTerm(),
        $search_modal = $('#mkdocs_search_modal');

    if(search_term){
        $search_modal.modal();
    }

    // make sure search input gets autofocus everytime modal opens.
    $search_modal.on('shown.bs.modal', function () {
        $search_modal.find('#mkdocs-search-query').focus();
    });

    // Highlight.js
    hljs.initHighlightingOnLoad();
    $('table').addClass('table table-striped table-hover');

    $('body').scrollspy({
        target: 'div.sidebar',
    });

    /* Prevent disabled links from causing a page reload */
    $("li.disabled a").click(function() {
        event.preventDefault();
    });

    // Manually hyphenate the sidebar
    $.get("/js/hypher-en-us.json").then(function (data) {
        window["Hypher"]["languages"]["en-us"] = new Hypher(data);
        $(".sidebar a").hyphenate("en-us");
    });

    // Change all span elements that have data-figureref to references to a
    // figure
    var reffrom = $("span[data-figureref]");
    var refto = reffrom.map(function () {
        return this.getAttribute("data-figureref")
    }).get();
    $("div[role='main'] figure").each(function (index) {
        // if the figure has a defined id
        if (this.id != "") {
            // search the figure in the references
            var ii = refto.indexOf(this.id);
            // if found replace the reference with the string Figure index+1
            // and a link to that figure
            if (ii >= 0) {
                reffrom[ii].innerHTML =
                    "<a href=\"#" +
                    this.id +
                    "\">Figure " +
                    (index+1) +
                    "</a>";
            }
        }
    });
});

