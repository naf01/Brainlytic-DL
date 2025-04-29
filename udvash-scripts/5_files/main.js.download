$(document).ready(function () {   
    /*From Jahangir For Mobile Menu*/

    $(document).click(function (event) {
        var icon = $('.icon');
        var eL = event.target;
        var menu = $('div.uu-mobile-menu');
      
        if (eL.id == 'wrapper' || eL.classList.contains('icon') || eL.classList.contains('line')) {
            event.preventDefault();
            icon.toggleClass('close');
            menu.toggleClass('uu-active');
            $('.uu-desible-body').toggleClass('show');
            var headerHeight = $('.uu-header-wrapper').outerHeight();
            $('.uuu-left-sidebar.uu-active, .uuu-left-sidebar.uu-active ul').css({ 'top': headerHeight });
        } else if (menu.is(eL) || menu.has(eL).length > 0) {
            if (eL.id == 'top-menu-wrap') {
                icon.removeClass('close');
                menu.removeClass('uu-active');
                $('.uu-desible-body').removeClass('show');
            } else {
                return true;
            }
        } else {
            icon.removeClass('close');
            menu.removeClass('uu-active');
            $('.uu-desible-body').removeClass('show');
        }
    });


    $('ul.uu-main-menu li a').click(function (event) {
        if (jQuery('body').width() <= 992) {
            if ($(this).attr('href') == '#' || $(this).attr('href') == '') {
                event.preventDefault();
                $(this).parent().first().toggleClass('open');
                $(this).parent().siblings().removeClass('open');
                $(this).parent().find('.open').removeClass('open');
            } else {
                return true;
            }
        }
    });

    /*== menu script ==*/
    $(document).on("click", "li.uu-profile", function (e) {
        e.stopPropagation();
        $(this).find(".uu-dropdown").slideToggle('fast');
    });

    $(document).on("click", ".uu-dropdown, ul.uu-routine-dropdown", function (e) {
        e.stopPropagation();
    });

    $(".uu-has-sub-menu").click(function (e) {
        e.stopPropagation();
    });



    $(document).on("click", "body", function () {
        $(".uu-dropdown, ul.uu-routine-dropdown").fadeOut('fast');
    });

    /*== Routine List Script ==*/
    $(document).on("click", "div.routine-lists > ul > li", function (e) {
        e.stopPropagation();

        $('div.routine-lists > ul > li').not($(this)).find("ul.uu-routine-dropdown").slideUp('fast');

        $(this).find("ul.uu-routine-dropdown").slideToggle('fast');
    });


    /*== Show bKash Info Script ==*/
    $(document).on("click", "a.uu-bkash-payment-with-tr-id", function (e) {
        $(this).addClass('uu-active');
        $(this).parent().find(".uu-bKash-payment-info").slideDown('fast');
    });
    $(document).on("click", ".uu-bKash-payment-info .uu-btn-cancel", function (e) {
        $('.uu-bkash-payment-with-tr-id').removeClass('uu-active');
        $(".uu-bKash-payment-info").slideUp('fast');
    });
    $(document).on("click", ".uu-bKash-payment-info .uu-btn-confirm", function (e) {
        $(".uu-confirm-wrapper").addClass('uu-confirm-active');
        $("body").css('overflow', 'hidden');
    });
    $(document).on("click", ".uu-btn-no", function (e) {
        $(".uu-confirm-wrapper").removeClass('uu-confirm-active');
        $("body").css('overflow', 'inherit');
    });

    //datepicker
    $('.datepicker').datepicker();

    //Redesign new js start
    var uuuSelectMenuType = getCookie("uuuSelectMenuType");
    if (uuuSelectMenuType == 'only-icon') {
        $('.uuu-left-sidebar').addClass('uuu-icon-sidebar');
        $('.uuu-left-sidebar .menu-text').addClass('d-none');
        $('.uuu-main-body').addClass('uuu-with-icon-main-body');
    } else {
        $('.uuu-left-sidebar').removeClass('uuu-icon-sidebar');
        $('.uuu-left-sidebar .menu-text').removeClass('d-none');
        $('.uuu-main-body').removeClass('uuu-with-icon-main-body');
    }
    $(document).on("click", ".uuu-desktop-burger", function (e) {
        $('.uuu-left-sidebar').toggleClass('uuu-icon-sidebar');
        $('.uuu-left-sidebar .menu-text').toggleClass('d-none');
        $('.uuu-main-body').toggleClass('uuu-with-icon-main-body');
        if ($('.uuu-left-sidebar').hasClass('uuu-icon-sidebar')) {
            setCookie("uuuSelectMenuType", "only-icon", 1);
        }
        else {
            setCookie("uuuSelectMenuType", "with-icon", 2);
        }
    });
    $(document).on("click", ".uuu-dashboard .uuu-fb-overlay-modal .fb-modal-close", function (e) {
        $(this).parents('.uuu-fb-overlay-modal').addClass('d-none');
    });
    function handleMenu() {
        var courseDynamicMenu = $('.uuu-new-menu > li').length;
        if ($(window).width() >= 991 && courseDynamicMenu > 6) {
            var clonedItems = $('.uuu-new-menu > li:gt(5)').clone();
            $('.uuu-new-menu > li:gt(5)').remove();
            $('.uuu-new-menu').append(`
        <li class="uu-course-dynamic-menu uu-has-sub-menu">
          <a href="#">More <i class="fas fa-chevron-down"></i></a>
          <ul class="more-menu uu-dropdown-menu"></ul>
        </li>
      `);
            $('.uuu-new-menu .more-menu').append(clonedItems);
        }
    }
    handleMenu();
    $(window).on('resize', function () {
        handleMenu();
    });

    ActiveMenuHeighlight();
    function ActiveMenuHeighlight() {
        var currentUrl = window.location.href;
        $('.uuu-left-sidebar ul li a').each(function () {
            var anchorHref = $(this).attr('href');
            if (currentUrl.indexOf(anchorHref) !== -1) {
                $(this).addClass('active');
            }            
        });
    }
    //Redesign new js end
});
function setCookie(name, value, days) {
    var expires = "";
    if (days) {
        var date = new Date();
        date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
        expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + value + expires + "; path=/";
}
function setCookieInMinute(name, value, minute) {
    var expires = "";
    if (minute) {
        var date = new Date();
        date.setTime(date.getTime() + (minute * 60 * 1000));
        expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + value + expires + "; path=/";
}
function getCookie(name) {
    var cookieArray = document.cookie.split(';');
    for (var i = 0; i < cookieArray.length; i++) {
        var cookie = cookieArray[i].trim();
        if (cookie.indexOf(name + "=") == 0) {
            return cookie.substring(name.length + 1, cookie.length);
        }
    }
    return null;
} 
function clearCookie(name) {
    document.cookie = name + "=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
}
